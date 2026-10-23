# Copyright 2026 Julien Blanchon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Batched Omniverse RTX (`ovrtx`) rendering of a MuJoCo scene: one camera per world, one step.

Newton's ViewerRTX builds the RTX stage (USD exported from the Newton model, its "studio"
lights and UsdPreviewSurface materials) for one world. Its render product is then turned into
IsaacLab's tiled sensor layout (isaaclab_ov OVRTX renderer):

- the world's shape prims are copied once per extra world in the USD layer, at the same place;
  each world gets its own scene partition (``primvars:omni:scenePartition = "env_<i>"``), and so
  does its camera (``omni:scenePartition``), so camera i only sees world i;
- the render product targets all cameras and its resolution is the tile atlas
  (ceil(sqrt(n)) columns): RTX renders every camera in one ``step``;
- the LdrColor atlas is mapped on the GPU (DLPack, stream-ordered, no host round trip) and split
  into per-world images by one Warp kernel.

``layout="products"`` instead gives every camera its own render product (all stepped together),
for comparison. ``mode="minimal"`` selects RTX's Minimal mode (rasterisation-like simple
shading; ``minimal_mode`` 1: constant diffuse, 2: diffuse MDL, 3: full MDL) instead of
RealTimePathTracing (1 spp, no denoiser, the viewer's settings).

The scene is static: world 0's prims keep the viewer's per-frame transform updates, the copies
do not (``update()`` only refreshes world 0). Requires an RTX GPU and ``newton[rtx]``.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import warp as wp

from references.newton_camera import newton_scene

if TYPE_CHECKING:
  from _common import RGB


@wp.kernel
def _extract_tiles(atlas: wp.array2d[wp.vec4ub], cols: int, first: int, res: int, out: wp.array4d[wp.uint8]):
  w, y, x = wp.tid()
  tile = first + w
  p = atlas[(w // cols) * res + y, (w % cols) * res + x]
  out[tile, y, x, 0] = p[0]
  out[tile, y, x, 1] = p[1]
  out[tile, y, x, 2] = p[2]


def _camera_matrix(
  pos: np.ndarray[[3], np.dtype[Any]], xmat: np.ndarray[[9], np.dtype[Any]]
) -> np.ndarray[[4, 4], np.dtype[np.float64]]:
  """USD row-vector world matrix of a MuJoCo camera (both look along -Z with +Y up)."""
  mat = np.eye(4)
  mat[:3, :3] = xmat.reshape(3, 3).T
  mat[3, :3] = pos
  return mat


def _tiled_viewer_class():
  from newton.viewer import ViewerRTX
  from pxr import Gf
  from pxr import Sdf
  from pxr import UsdGeom

  class TiledViewerRTX(ViewerRTX):
    """ViewerRTX whose stage holds `nworld` partitioned copies of the world and one camera each."""

    def configure(self, nworld, res, cam_pos, cam_xmat, fovy, mode, minimal_mode, shadows, layout, shift):
      self._tile = {
        "nworld": nworld,
        "res": res,
        "cam_pos": cam_pos,
        "cam_xmat": cam_xmat,
        "fovy": fovy,
        "mode": mode,
        "minimal_mode": minimal_mode,
        "shadows": shadows,
        "layout": layout,
        "shift": shift,
      }

    def _add_camera_lights_and_render_product(self):
      super()._add_camera_lights_and_render_product()
      t = self._tile
      n, res = t["nworld"], t["res"]
      stage = self.stage
      layer = stage.GetRootLayer()
      # World copies (Sdf-level, in C++) next to each world-0 shape batch, partitioned per world.
      name_to_shapes = {s.name: s for s in self._shape_instances.values()}
      batches = []
      for name, paths in self._instance_prim_paths.items():
        shapes = name_to_shapes.get(name)
        if shapes is None or not paths:
          continue
        worlds = shapes.worlds.numpy() if hasattr(shapes.worlds, "numpy") else np.asarray(shapes.worlds)
        if np.all(worlds < 0):
          continue  # global shapes: unpartitioned, seen by every camera
        batches.append((self._get_path(name), worlds))
      token = Sdf.ValueTypeNames.Token
      with Sdf.ChangeBlock():
        for batch, worlds in batches:
          src = Sdf.Path(batch)
          for w in reversed(range(n)):  # copies first: world 0 then gets its own partition
            dst = src if w == 0 else src.GetParentPath().AppendChild(f"{src.name}_world{w}")
            if w > 0:
              Sdf.CopySpec(layer, src, layer, dst)
            spec = layer.GetPrimAtPath(dst)
            attr = Sdf.AttributeSpec(spec, "primvars:omni:scenePartition", token)
            attr.default = f"env_{w}"
            if w > 0 and np.any(t["shift"]):
              # Debug: move world copies so partition leaks would show up as ghost images.
              spec.typeName = "Xform"
              op = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
              op.default = Gf.Vec3d(*(np.asarray(t["shift"], dtype=float) * w))
              order = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, variability=Sdf.VariabilityUniform)
              order.default = ["xformOp:translate"]
            for i in np.nonzero(worlds < 0)[0]:  # a global shape inside a world batch: keep one copy
              inst = layer.GetPrimAtPath(dst.AppendChild(f"instance_{i}"))
              if inst is not None and w > 0:
                inst.active = False
      # One camera per world, in that world's partition.
      v_aperture = 20.955
      focal = v_aperture / (2.0 * math.tan(math.radians(t["fovy"]) / 2.0))
      parent = Sdf.Path(self._camera_prim_path).GetParentPath()
      self._tile_cameras = []
      for w in range(n):
        path = str(parent.AppendChild(f"TileCamera_{w}"))
        cam = UsdGeom.Camera.Define(stage, path)
        cam.GetFocalLengthAttr().Set(focal)
        cam.GetHorizontalApertureAttr().Set(v_aperture)
        cam.GetVerticalApertureAttr().Set(v_aperture)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        xf = UsdGeom.Xform(cam.GetPrim())
        xf.ClearXformOpOrder()
        pos = np.asarray(t["cam_pos"], dtype=float) + np.asarray(t["shift"], dtype=float) * w
        xf.AddTransformOp().Set(Gf.Matrix4d(*_camera_matrix(pos, t["cam_xmat"]).flatten().tolist()))
        cam.GetPrim().CreateAttribute("omni:scenePartition", token).Set(f"env_{w}")
        self._tile_cameras.append(path)

      def configure_product(rp, cameras, width, height):
        rp.GetRelationship("camera").SetTargets([Sdf.Path(c) for c in cameras])
        rp.GetAttribute("resolution").Set(Gf.Vec2i(width, height))
        if t["mode"] == "minimal":
          rp.GetAttribute("omni:rtx:rendermode").Set("Minimal")
          rp.CreateAttribute("omni:rtx:minimal:mode", Sdf.ValueTypeNames.Int).Set(int(t["minimal_mode"]))
          rp.CreateAttribute("omni:rtx:minimal:castShadows", Sdf.ValueTypeNames.Bool).Set(bool(t["shadows"]))
        # As IsaacLab: ovrtx 0.5 eco mode freezes the output; wait for loading on the first step.
        rp.CreateAttribute("omni:rtx:rt:ecoMode:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.GetAttribute("omni:rtx:waitForEvents").Set(["AllLoadingFinished", "OnlyOnFirstRequest"])

      viewer_product = stage.GetPrimAtPath(self._render_product_path)
      if t["layout"] == "tiled":
        self.tile_cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / self.tile_cols)
        configure_product(viewer_product, self._tile_cameras, self.tile_cols * res, rows * res)
        self.tile_products = [self._render_product_path]
      else:  # one product per camera
        self.tile_cols = 1
        self.tile_products = []
        scope = Sdf.Path(self._render_product_path).GetParentPath()
        for w, cam in enumerate(self._tile_cameras):
          path = scope.AppendChild(f"TileProduct_{w}")
          Sdf.CopySpec(layer, Sdf.Path(self._render_product_path), layer, path)
          configure_product(stage.GetPrimAtPath(path), [cam], res, res)
          self.tile_products.append(str(path))
        settings = stage.GetPrimAtPath("/Render/OmniverseGlobalRenderSettings")
        if settings:
          settings.GetRelationship("products").SetTargets([Sdf.Path(p) for p in self.tile_products])

  return TiledViewerRTX


class OvrtxTiledCamera:
  """`nworld` copies of a MuJoCo scene rendered by Omniverse RTX from the MuJoCo camera "cam"."""

  rgb: wp.array4d[wp.uint8]  # (nworld, res, res, 3), the last frame of every world

  def __init__(
    self,
    spec: mujoco.MjSpec,
    mjm: mujoco.MjModel,
    nworld: int,
    res: int,
    mode: str = "rtpt",
    minimal_mode: int = 3,
    shadows: bool = True,
    layout: str = "tiled",
    environment: str = "studio",
    pipelined: bool = False,
    shift=(0.0, 0.0, 0.0),
  ):
    from ovrtx import Device

    self._device_cuda = Device.CUDA
    self.nworld, self.res, self.pipelined = nworld, res, pipelined
    self.model, self.state, _, fovy = newton_scene(spec, mjm, 1)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    self.viewer = _tiled_viewer_class()(width=res, height=res, headless=True, environment=environment, async_rendering=False)
    self.viewer.configure(
      nworld, res, mjd.cam_xpos[cam].copy(), mjd.cam_xmat[cam].copy(), fovy, mode, minimal_mode, shadows, layout, shift
    )
    self.viewer.set_model(self.model)
    self.viewer.camera.fov = fovy
    self.time = 0.0
    self.timings = {"setup_s": 0.0}
    start = time.perf_counter()
    self.update()  # first frame: exports the stage and loads it into RTX (one viewer step)
    self.timings["setup_s"] = time.perf_counter() - start
    self.rgb = wp.zeros((nworld, res, res, 3), dtype=wp.uint8)
    self._pending = None

  def update(self):
    """Scene update: world-0 shape transforms written to the RTX stage (synchronous)."""
    self.time += 1.0 / 60.0
    v = self.viewer
    v.begin_frame(self.time)
    v.log_state(self.state)
    if v._phase == v._PHASE_BUILD:
      v.end_frame()  # builds and loads the stage, then renders once through the viewer
    else:
      v._update_scene()

  def _step_kwargs(self):
    kwargs = {"render_products": set(self.viewer.tile_products), "delta_time": 1.0 / 60.0}
    if self.viewer._use_ovstage:
      kwargs["ordinal"] = self.viewer._ovstage_ordinal
    return kwargs

  def step(self):
    """RTX step of every camera (blocks until rendered); returns the render products."""
    return self.viewer._rtx.step(**self._step_kwargs())

  def readback(self, products):
    """Maps the LdrColor outputs and splits them into `self.rgb` (nworld, res, res, 3)."""
    stream = wp.get_stream()
    handle = stream.cuda_stream or 1
    first = 0
    for path in self.viewer.tile_products:
      frame = products[path].frames[0]
      var = self.viewer._get_ldr_color_render_var(frame)
      mapping = var.map(device=self._device_cuda, sync_stream=handle)
      try:
        atlas = wp.from_dlpack(mapping, dtype=wp.vec4ub)
        count = self.nworld if len(self.viewer.tile_products) == 1 else 1
        wp.launch(
          _extract_tiles,
          dim=(count, self.res, self.res),
          inputs=[atlas, self.viewer.tile_cols, first, self.res],
          outputs=[self.rgb],
        )
      finally:
        mapping.unmap(stream=handle)
      first += 1

  def render(self):
    """One batch: step + GPU readback (pipelined: submit the next step, then read the last one)."""
    if not self.pipelined:
      self.readback(self.step())
      return
    op = self.viewer._rtx.step_async(**self._step_kwargs())
    previous, self._pending = self._pending, op
    if previous is not None:
      self.readback(previous.wait().fetch())

  def profile(self, iters: int = 20) -> dict:
    """Median milliseconds of update / step / readback, each synchronised."""
    t = {"update_ms": [], "step_ms": [], "readback_ms": []}
    for _ in range(iters):
      wp.synchronize()
      a = time.perf_counter()
      self.update()
      wp.synchronize()
      b = time.perf_counter()
      products = self.step()
      c = time.perf_counter()
      self.readback(products)
      wp.synchronize()
      d = time.perf_counter()
      t["update_ms"].append((b - a) * 1e3)
      t["step_ms"].append((c - b) * 1e3)
      t["readback_ms"].append((d - c) * 1e3)
    return {k: float(np.median(v)) for k, v in t.items()}

  def close(self):
    if self._pending is not None:
      self._pending.wait().fetch()
      self._pending = None
    self.viewer.close()

  def image(self, world: int = 0) -> RGB[int, int]:
    """Last frame of one world as float RGB in [0, 1], (res, res, 3)."""
    return self.rgb.numpy()[world].astype(np.float32) / 255.0

  def atlas(self, count: int = 16) -> np.ndarray[[int, int, 3], np.dtype[np.uint8]]:
    """The first `count` worlds' last frames side by side, uint8 (res, count * res, 3)."""
    imgs = self.rgb.numpy()[:count]
    return np.concatenate(list(imgs), axis=1)
