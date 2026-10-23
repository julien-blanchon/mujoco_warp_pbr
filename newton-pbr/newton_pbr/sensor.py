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

"""PBR camera sensor for Newton, mirroring `newton.sensors.SensorCamera.update`."""

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp

import mujoco_warp_pbr as mjpbr
from newton_pbr._scene import Light
from newton_pbr._scene import build_scene
from newton_pbr._scene import default_lights

_OBJ_GEOM = wp.constant(int(mujoco.mjtObj.mjOBJ_GEOM))


@wp.kernel(enable_backward=False)
def _sync_poses(
  body_q: wp.array[wp.transformf],
  shape_body: wp.array[wp.int32],
  shape_transform: wp.array[wp.transformf],
  geom_shape: wp.array2d[wp.int32],
  offset_pos: wp.array[wp.vec3],
  offset_quat: wp.array[wp.quat],
  world_of_view: wp.array[wp.int32],
  camera_transforms: wp.array[wp.transformf],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  cam_xpos: wp.array2d[wp.vec3],
  cam_xmat: wp.array2d[wp.mat33],
):
  """World-space geom poses of view v's world from Newton's body transforms; camera pose."""
  view, geom = wp.tid()
  world = world_of_view[view]
  shape = geom_shape[world, geom]
  x = shape_transform[shape]
  body = shape_body[shape]
  if body >= 0:
    x = wp.transform_multiply(body_q[body], x)
  x = wp.transform_multiply(x, wp.transform(offset_pos[geom], offset_quat[geom]))
  geom_xpos[view, geom] = wp.transform_get_translation(x)
  geom_xmat[view, geom] = wp.quat_to_matrix(wp.transform_get_rotation(x))
  if geom == 0:
    # Newton and MuJoCo cameras both look along -Z with +Y up.
    c = camera_transforms[view]
    cam_xpos[view, 0] = wp.transform_get_translation(c)
    cam_xmat[view, 0] = wp.quat_to_matrix(wp.transform_get_rotation(c))


@wp.kernel(enable_backward=False)
def _copy_outputs(
  width: int,
  rgb_in: wp.array2d[wp.uint32],
  depth_in: wp.array2d[float],
  seg_in: wp.array2d[wp.vec2i],
  hdr_in: wp.array2d[wp.vec3],
  geom_shape: wp.array2d[wp.int32],
  world_of_view: wp.array[wp.int32],
  ray_scale: wp.array2d[float],
  want_color: bool,
  want_depth: bool,
  want_forward_depth: bool,
  want_shape: bool,
  want_hdr: bool,
  color: wp.array3d[wp.uint32],
  depth: wp.array3d[float],
  forward_depth: wp.array3d[float],
  shape_index: wp.array3d[wp.uint32],
  hdr: wp.array3d[wp.vec3],
):
  view, pixel = wp.tid()
  y = pixel / width
  x = pixel - y * width
  if want_color:
    # MuJoCo Warp packs 0xAARRGGBB, Newton 0xAABBGGRR.
    c = rgb_in[view, pixel]
    color[view, y, x] = (
      (c & wp.uint32(0xFF00FF00)) | ((c >> wp.uint32(16)) & wp.uint32(0xFF)) | ((c & wp.uint32(0xFF)) << wp.uint32(16))
    )
  # Newton: negative depth for rays that hit nothing; MuJoCo Warp: 0.
  d = depth_in[view, pixel]
  if d <= 0.0:
    d = -1.0
  if want_forward_depth:
    forward_depth[view, y, x] = d
  if want_depth:
    depth[view, y, x] = wp.where(d > 0.0, d * ray_scale[y, x], d)
  if want_shape:
    s = seg_in[view, pixel]
    index = wp.uint32(0xFFFFFFFF)
    if s[0] >= 0 and s[1] == _OBJ_GEOM:
      index = wp.uint32(geom_shape[world_of_view[view], s[0]])
    shape_index[view, y, x] = index
  if want_hdr:
    hdr[view, y, x] = hdr_in[view, pixel]


class SensorPbrCamera:
  """Physically based camera sensor for a Newton model, rendered by mujoco_warp_pbr.

  The interface follows :class:`newton.sensors.SensorCamera`: one view per world, camera
  transforms place a camera that looks along its local -Z axis with +Y up, and outputs have
  shape ``(view_count, height, width)`` in Newton's formats (packed RGBA ``uint32`` colour,
  ray-distance and forward depth with -1 for no hit, shape indices with ``0xFFFFFFFF`` for no
  hit, linear HDR). Unlike ``SensorCamera``, the pinhole image size and field of view are fixed
  at construction instead of being passed as a ray bundle.

  Appearance comes from the Newton model: shape colours (sRGB) and opacities, mesh base-colour
  textures and scalar metallic / roughness. Newton models carry no lights: pass ``envmap`` for
  image-based lighting and / or ``lights``; without either, :func:`default_lights` are used.

  Example::

      camera = SensorPbrCamera(model, width=128, height=128, fovy=60.0, envmap="studio.hdr")
      color = camera.allocate_image_color()
      camera.update(state, camera_transforms, color_image=color)
  """

  Light = Light

  def __init__(
    self,
    model: newton.Model,
    *,
    width: int,
    height: int,
    fovy: float = 60.0,
    lights: list[Light] | None = None,
    envmap=None,
    headlight=None,
    **render_options,
  ):
    """Builds the MuJoCo appearance model and the render context.

    Args:
      model: Newton model; every world must have the same visible shapes.
      width: Image width [px].
      height: Image height [px].
      fovy: Vertical field of view [deg].
      lights: Lights added to the scene; None uses :func:`default_lights` when no envmap is given.
      envmap: Equirectangular HDR environment (path or array) lighting the scene.
      headlight: None (off) or (ambient, diffuse, specular) RGB of MuJoCo's camera headlight.
      **render_options: Further `mujoco_warp_pbr.create_render_context` options (e.g. `tier`,
        `samples_per_pixel`, `light_model`, `envmap_intensity`).
    """
    self.model = model
    self.width = width
    self.height = height
    if lights is None and envmap is None:
      lights = default_lights()
    self.scene = build_scene(model, lights, fovy, headlight)
    mjm = self.scene.mjm
    self.nworld = max(int(model.world_count), 1)
    self.mjm = mjm
    self.m = mjw.put_model(mjm)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    self.d = mjw.put_data(mjm, mjd, nworld=self.nworld)
    if envmap is not None:
      render_options.setdefault("envmap", envmap)
    render_options.setdefault("render_depth", True)
    render_options.setdefault("render_seg", True)
    self.rc = mjpbr.create_render_context(
      mjm, nworld=self.nworld, cam_res=(width, height), render_rgb=True, render_hdr=True, **render_options
    )
    device = model.device
    self._geom_shape = wp.array(self.scene.geom_shape, dtype=wp.int32, device=device)
    self._offset_pos = wp.array(self.scene.geom_offset_pos.astype(np.float32), dtype=wp.vec3, device=device)
    # MuJoCo quaternions are (w, x, y, z), Warp's (x, y, z, w).
    q = self.scene.geom_offset_quat[:, [1, 2, 3, 0]].astype(np.float32)
    self._offset_quat = wp.array(q, dtype=wp.quat, device=device)
    self._world_of_view = wp.array(np.arange(self.nworld, dtype=np.int32), dtype=wp.int32, device=device)
    # Ray distance / forward depth per pixel (pinhole through pixel centres).
    tan = np.tan(np.radians(fovy) / 2.0)
    ys = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * tan
    xs = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * tan * width / height
    scale = np.sqrt(1.0 + xs[None, :] ** 2 + ys[:, None] ** 2)
    self._ray_scale = wp.array(scale.astype(np.float32), dtype=float, device=device)

  def allocate_image_color(self) -> wp.array3d[wp.uint32]:
    """Packed RGBA colour image (Newton layout), shape (world_count, height, width)."""
    return wp.zeros((self.nworld, self.height, self.width), dtype=wp.uint32, device=self.model.device)

  def allocate_image_depth(self) -> wp.array3d[wp.float32]:
    """Depth image [m], -1 where nothing is hit, shape (world_count, height, width)."""
    return wp.zeros((self.nworld, self.height, self.width), dtype=wp.float32, device=self.model.device)

  def allocate_image_shape_index(self) -> wp.array3d[wp.uint32]:
    """Shape-index image, 0xFFFFFFFF where nothing is hit, shape (world_count, height, width)."""
    return wp.zeros((self.nworld, self.height, self.width), dtype=wp.uint32, device=self.model.device)

  def allocate_image_color_hdr(self) -> wp.array3d[wp.vec3f]:
    """Linear HDR radiance image (before exposure), shape (world_count, height, width)."""
    return wp.zeros((self.nworld, self.height, self.width), dtype=wp.vec3f, device=self.model.device)

  def update(
    self,
    state: newton.State,
    camera_transforms: wp.array[wp.transformf],
    *,
    color_image: wp.array3d[wp.uint32] | None = None,
    depth_image: wp.array3d[wp.float32] | None = None,
    forward_depth_image: wp.array3d[wp.float32] | None = None,
    shape_index_image: wp.array3d[wp.uint32] | None = None,
    hdr_color_image: wp.array3d[wp.vec3f] | None = None,
    seed: int = 0,
  ):
    """Renders view i = world i from camera_transforms[i] (shape (world_count,))."""
    if camera_transforms.shape[0] != self.nworld:
      raise ValueError(f"camera_transforms must have one transform per world ({self.nworld})")
    ngeom = self.mjm.ngeom
    wp.launch(
      _sync_poses,
      dim=(self.nworld, ngeom),
      inputs=[
        state.body_q,
        self.model.shape_body,
        self.model.shape_transform,
        self._geom_shape,
        self._offset_pos,
        self._offset_quat,
        self._world_of_view,
        camera_transforms,
      ],
      outputs=[self.d.geom_xpos, self.d.geom_xmat, self.d.cam_xpos, self.d.cam_xmat],
    )
    mjw.refit_bvh(self.m, self.d, self.rc.base)
    mjpbr.render(self.m, self.d, self.rc, seed=seed)

    base = self.rc.base
    dummy_u = wp.empty((1, 1, 1), dtype=wp.uint32, device=self.model.device)
    dummy_f = wp.empty((1, 1, 1), dtype=float, device=self.model.device)
    dummy_v = wp.empty((1, 1, 1), dtype=wp.vec3, device=self.model.device)
    wp.launch(
      _copy_outputs,
      dim=(self.nworld, self.width * self.height),
      inputs=[
        self.width,
        base.rgb_data,
        base.depth_data,
        base.seg_data,
        self.rc.hdr_data,
        self._geom_shape,
        self._world_of_view,
        self._ray_scale,
        color_image is not None,
        depth_image is not None,
        forward_depth_image is not None,
        shape_index_image is not None,
        hdr_color_image is not None,
      ],
      outputs=[
        color_image if color_image is not None else dummy_u,
        depth_image if depth_image is not None else dummy_f,
        forward_depth_image if forward_depth_image is not None else dummy_f,
        shape_index_image if shape_index_image is not None else dummy_u,
        hdr_color_image if hdr_color_image is not None else dummy_v,
      ],
    )
