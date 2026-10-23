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

"""ViewerPbr: Newton's GL viewer with the main view rendered by mujoco_warp_pbr."""

from __future__ import annotations

from typing import Any, cast, override

import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp
from newton._src.viewer.camera import Camera
from newton.viewer import ViewerBase
from newton.viewer import ViewerGL

import mujoco_warp_pbr as mjpbr
from newton_pbr._render import ScenePoses
from newton_pbr._render import put_data
from newton_pbr._scene import Headlight
from newton_pbr._scene import Light
from newton_pbr._scene import build_scene
from newton_pbr._scene import default_lights

# create_render_context options the viewer sets itself.
_RESERVED_OPTIONS = ("nworld", "cam_res", "cam_active", "render_rgb", "camera_rays")


@wp.kernel(enable_backward=False)
def _to_rgb8(width: int, rgb: wp.array2d[wp.uint32], out: wp.array3d[wp.uint8]):
  """MuJoCo Warp's packed colour (0xAARRGGBB) of world 0 as an (height, width, 3) uint8 image."""
  pixel = wp.tid()
  y = pixel / width
  x = pixel - y * width
  c = int(rgb[0, pixel])
  out[y, x, 0] = wp.uint8((c >> 16) & 0xFF)
  out[y, x, 1] = wp.uint8((c >> 8) & 0xFF)
  out[y, x, 2] = wp.uint8(c & 0xFF)


def _unit(v: Any) -> Any:
  return v / max(float(np.sqrt(np.dot(v, v))), 1e-12)


def _camera_basis(camera: Any) -> tuple[np.ndarray, np.ndarray]:
  """(position, rotation with columns right, up, back) of a Newton viewer camera."""
  front = np.array([float(v) for v in camera.get_front()][:3])
  up = np.array([float(v) for v in camera.get_up()][:3])
  right = _unit(np.cross(front, up))
  up = np.cross(right, front)
  pos = np.array([float(v) for v in camera.pos][:3])
  return pos, np.stack([right, up, -front], axis=1)


class _PbrFrames:
  """mujoco_warp_pbr renders of a Newton model (all worlds in one scene) from a viewer camera."""

  def __init__(self, lights: list[Light] | None, envmap: Any, headlight: Headlight | None, options: dict[str, Any]):
    reserved = [name for name in _RESERVED_OPTIONS if name in options]
    if reserved:
      raise TypeError(f"ViewerPbr sets {reserved} itself; remove them from the render options")
    self.lights = lights if lights is not None or envmap is not None else default_lights()
    self.headlight = headlight
    self.options = dict(options)
    if envmap is not None:
      self.options.setdefault("envmap", envmap)
    self.model: newton.Model | None = None
    self._scene: tuple[Any, mjw.Model, ScenePoses] | None = None
    self._view: tuple[tuple[int, int], mjw.Data, mjpbr.PbrRenderContext[int], wp.array3d[wp.uint8]] | None = None
    self.frame: wp.array3d[wp.uint8] | None = None

  def set_model(self, model: newton.Model | None) -> None:
    self.model = model
    self._scene = None
    self._view = None
    self.frame = None
    if model is None:
      return
    scene = build_scene(model, self.lights, 45.0, self.headlight, merge_worlds=True)
    if scene.mjm.ngeom == 0:
      return
    with wp.ScopedDevice(model.device):
      m = mjw.put_model(scene.mjm)
    self._scene = (scene.mjm, m, ScenePoses(model, scene))

  def render(
    self, state: newton.State, camera: Any, width: int, height: int, world_offsets: wp.array1d[wp.vec3] | None, seed: int
  ) -> wp.array3d[wp.uint8] | None:
    """Renders the scene from a Newton viewer camera; None for a model without visible shapes."""
    if self._scene is None or self.model is None:
      return None
    mjm, m, poses = self._scene
    device = self.model.device
    size = (max(int(width), 1), max(int(height), 1))
    if self._view is None or self._view[0] != size:
      self._view = None  # release the previous resolution first
      d = put_data(mjm, 1, device)
      with wp.ScopedDevice(device):
        rc = mjpbr.create_render_context(mjm, nworld=1, cam_res=size, render_rgb=True, **self.options)
        frame = wp.zeros((size[1], size[0], 3), dtype=wp.uint8)
      self._view = (size, d, rc, frame)
    _, d, rc, frame = self._view
    pos, rot = _camera_basis(camera)
    xform = wp.transformf(wp.vec3f(*pos), wp.quat_from_matrix(wp.mat33f(*rot.astype(np.float32).ravel())))
    cams = wp.array([xform], dtype=wp.transformf, device=device)
    m.cam_fovy.fill_(float(camera.fov))
    rc.base.zfar = float(camera.far)
    poses.sync(state, cams, d, world_offsets=world_offsets)
    if rc.traversal != "optix":  # OptiX updates its own structures in render
      mjw.refit_bvh(m, d, rc.base)
    mjpbr.render(m, d, rc, seed=seed)
    wp.launch(_to_rgb8, dim=size[0] * size[1], inputs=[size[0], rc.base.rgb_data], outputs=[frame], device=device)
    self.frame = frame
    return frame


class _HeadlessCamera(Camera):
  """Newton's viewer camera without pyglet: NumPy vectors, same yaw / pitch conventions."""

  def __init__(
    self, fov: float = 45.0, near: float = 0.01, far: float = 1000.0, width: int = 1280, height: int = 720, up_axis: Any = "Z"
  ):
    self.fov = fov
    self.near = near
    self.far = far
    self.width = width
    self.height = height
    self.up_axis = up_axis if isinstance(up_axis, int) else "XYZ".index(str(up_axis).upper())
    self.pos = self._as_vec3(self.get_default_position(self.up_axis))
    self.pitch = self.DEFAULT_PITCH
    self.yaw = self.DEFAULT_YAW
    self.pivot = self.pos + self.get_front() * self.DEFAULT_PIVOT_DISTANCE

  @staticmethod
  def _as_vec3(value: Any) -> Any:
    return np.array([float(value[0]), float(value[1]), float(value[2])])

  def _set_orientation_from_direction(self, direction: Any) -> None:
    d = _unit(self._as_vec3(direction))
    vertical = d[self.up_axis]
    horizontal = {0: (d[2], d[1]), 1: (d[2], d[0]), 2: (d[1], d[0])}[self.up_axis]
    self.pitch = self._clamp_pitch(np.rad2deg(np.arcsin(np.clip(vertical, -1.0, 1.0))))
    self.yaw = self._wrap_yaw(np.rad2deg(np.arctan2(*horizontal)))

  def get_front(self) -> Any:
    pitch = np.deg2rad(self._clamp_pitch(self.pitch))
    yaw = np.deg2rad(self.yaw)
    level = np.cos(pitch)
    if self.up_axis == 0:
      front = (np.sin(pitch), np.cos(yaw) * level, np.sin(yaw) * level)
    elif self.up_axis == 2:
      front = (np.cos(yaw) * level, np.sin(yaw) * level, np.sin(pitch))
    else:
      front = (np.cos(yaw) * level, np.sin(pitch), np.sin(yaw) * level)
    return _unit(np.array(front))

  def get_up(self) -> Any:
    world_up = np.zeros(3)
    world_up[self.up_axis] = 1.0
    front = self.get_front()
    right = _unit(np.cross(front, world_up))
    return _unit(np.cross(right, front))

  def get_right(self) -> Any:
    return _unit(np.cross(self.get_front(), self.get_up()))


class ViewerPbr(ViewerGL):
  """Newton's :class:`~newton.viewer.ViewerGL` with the main view rendered by mujoco_warp_pbr.

  Every frame the scene is path- or ray-traced from the viewer's current camera (its pose,
  vertical field of view and framebuffer size, scaled by ``render_scale``) and shown through
  ViewerGL's fullscreen image path, so the window, ImGui panels, camera controls (WASD / mouse),
  picking and pausing keep working. Switch with one constructor::

      viewer = newton_pbr.ViewerPbr()  # instead of: newton.viewer.ViewerGL()

  What still comes from GL: the UI panels and logged images. Debug overlays (contacts, joints,
  centres of mass, logged lines / points / meshes, gizmos), particles and deformables are drawn
  by the GL renderer only: untick "PBR (mujoco_warp_pbr)" in the Rendering panel, or set
  :attr:`pbr_enabled` to False, to see them. All worlds are rendered, at the viewer's world
  offsets; ``set_visible_worlds`` only affects the GL view.

  With ``headless=True`` no window or OpenGL context is created (nothing GL-related is needed):
  :meth:`get_frame` returns the PBR frame of the last :meth:`end_frame`, at ``width`` x
  ``height``, e.g. for recording videos; :meth:`is_running` stops after ``num_frames``.

  Attributes:
    pbr_enabled: Render the main view with mujoco_warp_pbr (else ViewerGL's GL renderer).
    render_scale: Render resolution relative to the window's framebuffer (windowed mode).
    seed: Random seed of the next frame (soft shadows, path tracing).
  """

  def __init__(
    self,
    width: int = 1920,
    height: int = 1080,
    vsync: bool = False,
    headless: bool = False,
    paused: bool = False,
    plot_history_size: int = 250,
    num_frames: int | None = None,
    *,
    enable_cuda_interop: ViewerGL.CudaInterop = ViewerGL.CudaInterop.DYNAMIC_MESH,
    lights: list[Light] | None = None,
    envmap: Any = None,
    headlight: Headlight | None = None,
    render_scale: float = 1.0,
    **render_options: Any,
  ):
    """Create the viewer (same arguments as ViewerGL, plus the PBR options).

    Args:
      width: Window width [px] (headless: PBR frame width).
      height: Window height [px] (headless: PBR frame height).
      vsync: Enable vertical sync.
      headless: No window and no OpenGL context; :meth:`get_frame` returns the PBR frame.
      paused: Start the viewer in paused mode.
      plot_history_size: Samples kept per :meth:`log_scalar` signal.
      num_frames: Frames before :meth:`is_running` returns False in headless mode.
      enable_cuda_interop: ViewerGL's CUDA-OpenGL interoperability categories.
      lights: Lights of the scene; None uses :func:`newton_pbr.default_lights` unless
        ``envmap`` is given.
      envmap: Equirectangular HDR environment (path or array) for image-based lighting.
      headlight: None (off) or (ambient, diffuse, specular) RGB of MuJoCo's headlight.
      render_scale: PBR resolution relative to the framebuffer (windowed mode).
      **render_options: Further :func:`mujoco_warp_pbr.create_render_context` options (e.g.
        ``tier``, ``samples_per_pixel``, ``traversal``).
    """
    self._gl_free = headless
    self._pbr = _PbrFrames(lights, envmap, headlight, render_options)
    self.pbr_enabled = True
    self.render_scale = float(render_scale)
    self.seed = 0
    if not headless:
      super().__init__(
        width, height, vsync, headless, paused, plot_history_size, num_frames, enable_cuda_interop=enable_cuda_interop
      )
      return
    if num_frames is not None and (not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 0):
      raise ValueError("num_frames must be an integer >= 0 or None")
    ViewerBase.__init__(self)
    self.camera = _HeadlessCamera(width=width, height=height, up_axis="Z")
    self.gui = None
    self.show_ui = False
    self._headless = True
    self.num_frames = num_frames
    self._frame_count = 0
    self._paused = paused
    self._step_requested = False
    self._reset_callback = None
    self._gizmo_log = None
    self.gizmo_is_using = False

  # Model and state.

  @override
  def clear_model(self):
    if self.__dict__.get("_gl_free"):
      ViewerBase.clear_model(self)
      self.picking = None
      self.wind = None
      self._last_state = None
    else:
      super().clear_model()
    pbr = self.__dict__.get("_pbr")
    if pbr is not None:
      pbr.set_model(None)

  @override
  def set_model(self, model: newton.Model | None):
    if self._gl_free:
      prev = self.camera
      ViewerBase.set_model(self, model)
      if model is not None and int(newton.Axis.from_any(model.up_axis)) != prev.up_axis:
        self.camera = _HeadlessCamera(width=prev.width, height=prev.height, up_axis=int(newton.Axis.from_any(model.up_axis)))
    else:
      super().set_model(model)
    self._pbr.set_model(model)

  @override
  def set_visible_worlds(self, worlds: Any) -> None:
    if self._gl_free:
      ViewerBase.set_visible_worlds(self, worlds)
    else:
      super().set_visible_worlds(worlds)

  @override
  def log_state(self, state: newton.State):
    if self._gl_free:
      self._last_state = state
    else:
      super().log_state(state)

  # Frames.

  def _render_size(self) -> tuple[int, int]:
    if self._gl_free:
      return int(self.camera.width), int(self.camera.height)
    return round(self.camera.width * self.render_scale), round(self.camera.height * self.render_scale)

  def render_pbr(self) -> wp.array3d[wp.uint8] | None:
    """Renders the PBR frame of the last logged state from the current camera (or None)."""
    state = self.__dict__.get("_last_state")
    if state is None or self.model is None:
      return None
    width, height = self._render_size()
    # Newton's world_offsets is a (world_count,) vec3 array (typed (int, 3) from its NumPy source).
    offsets = cast("wp.array1d[wp.vec3] | None", self.world_offsets)
    return self._pbr.render(state, self.camera, width, height, offsets, self.seed)

  @override
  def end_frame(self):
    if self.pbr_enabled and not self.is_rendering_paused():
      frame = self.render_pbr()
      if frame is not None and not self._gl_free:
        self.log_image("pbr", frame, fullscreen=True)
    if self._gl_free:
      self._frame_count += 1
    else:
      super().end_frame()

  @override
  def get_frame(
    self,
    *,
    output: wp.array3d[wp.uint8] | None = None,
    render_ui: bool = False,
    target_image: wp.array3d[wp.uint8] | None = None,
  ) -> wp.array3d[wp.uint8]:
    """The last frame, (height, width, 3) uint8 with a top-left origin.

    The PBR frame (render resolution) in headless mode, or in windowed mode while
    :attr:`pbr_enabled` without ``render_ui``; else ViewerGL's framebuffer read-back.

    Raises:
      RuntimeError: when no PBR frame was rendered yet (headless).
      ValueError: for an output of another shape.
    """
    if not self._gl_free and (not self.pbr_enabled or render_ui or target_image is not None):
      return super().get_frame(output=output, render_ui=render_ui, target_image=target_image)
    frame = self._pbr.frame
    if frame is None:
      if self._gl_free:
        raise RuntimeError("ViewerPbr.get_frame: no frame rendered yet (log_state and end_frame first)")
      return super().get_frame(output=output, render_ui=render_ui)
    if output is None:
      return wp.clone(frame)
    if tuple(output.shape) != tuple(frame.shape):
      raise ValueError(f"Shape of `output` must be {tuple(frame.shape)}, got {tuple(output.shape)}")
    wp.copy(output, frame)
    return output

  @override
  def is_running(self) -> bool:
    if self._gl_free:
      return self.num_frames is None or self._frame_count < self.num_frames
    return super().is_running()

  @override
  def close(self):
    self._pbr.set_model(None)
    if not self._gl_free:
      super().close()

  @override
  def _ui_populate_rendering_panel(self, imgui):
    _changed, self.pbr_enabled = imgui.checkbox("PBR (mujoco_warp_pbr)", self.pbr_enabled)
    super()._ui_populate_rendering_panel(imgui)

  # GL-only logging and input: no-ops without a window.

  @override
  def apply_forces(self, state: newton.State):
    if not self._gl_free:
      super().apply_forces(state)

  @override
  def log_mesh(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_mesh(*args, **kwargs)

  @override
  def log_instances(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_instances(*args, **kwargs)

  @override
  def log_capsules(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_capsules(*args, **kwargs)

  @override
  def log_lines(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_lines(*args, **kwargs)

  @override
  def log_arrows(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_arrows(*args, **kwargs)

  @override
  def log_wireframe_shape(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_wireframe_shape(*args, **kwargs)

  @override
  def clear_wireframe_vbo_cache(self) -> None:
    if not self._gl_free:
      super().clear_wireframe_vbo_cache()

  @override
  def log_points(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_points(*args, **kwargs)

  @override
  def log_gaussian(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_gaussian(*args, **kwargs)

  @override
  def log_gizmo(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_gizmo(*args, **kwargs)

  @override
  def log_array(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_array(*args, **kwargs)

  @override
  def log_scalar(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_scalar(*args, **kwargs)

  @override
  def log_image(self, *args: Any, **kwargs: Any) -> None:
    if not self._gl_free:
      super().log_image(*args, **kwargs)

  @override
  def is_key_down(self, key: str | int) -> bool:
    return False if self._gl_free else super().is_key_down(key)

  @property
  @override
  def vsync(self) -> bool:
    return False if self._gl_free else self.renderer.get_vsync()

  @vsync.setter
  def vsync(self, enabled: bool):
    if not self._gl_free:
      self.renderer.set_vsync(enabled)
