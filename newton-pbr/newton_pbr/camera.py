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

"""Drop-in replacement for `newton.sensors.SensorCamera`, rendered by mujoco_warp_pbr.

Each view is one render world of a mujoco_warp_pbr context whose geoms take the poses of the
view's Newton world, so views can show any world (several views per world, stereo pairs, ...).
The view's camera is rendered from Newton's camera-space ray bundle (mujoco_warp_pbr's
ray-bundle mode); outputs are converted to Newton's formats.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from typing import Any

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera as NewtonSensorCamera
from newton.utils import ColorSpace

import mujoco_warp_pbr as mjpbr
from newton_pbr._render import ScenePoses
from newton_pbr._render import put_data
from newton_pbr._scene import Checkerboard
from newton_pbr._scene import Headlight
from newton_pbr._scene import Light
from newton_pbr._scene import build_scene
from newton_pbr._scene import default_lights

ClearData = NewtonSensorCamera.ClearData
RenderConfig = NewtonSensorCamera.RenderConfig
AntiAliasing = NewtonSensorCamera.AntiAliasing
RenderOrder = NewtonSensorCamera.RenderOrder
WorldRenderFlag = NewtonSensorCamera.WorldRenderFlag
TextureProjectionMode = NewtonSensorCamera.TextureProjectionMode

_OBJ_GEOM = wp.constant(int(mujoco.mjtObj.mjOBJ_GEOM))
_DISABLE_PRESERVE = wp.constant(int(WorldRenderFlag.DISABLE_PRESERVE))
# Typed constants: a bare literal above INT32_MAX would first be emitted as an int32 by Warp.
_ALPHA_GREEN = wp.constant(wp.uint32(0xFF00FF00))
_BYTE = wp.constant(wp.uint32(0xFF))
# create_render_context options the sensor sets itself.
_RESERVED_OPTIONS = (
  "nworld",
  "cam_res",
  "cam_active",
  "render_rgb",
  "render_depth",
  "render_seg",
  "render_hdr",
  "render_guides",
  "camera_rays",
)
# Placeholder vertical field of view of the scene's camera (rays replace its projection).
_FOVY = 60.0
_DEFAULT_CONFIG = RenderConfig()
# Newton's default light: travelling along normalised (-1, 1, -1).
_DEFAULT_LIGHT_DIRECTION = (-0.57735026, 0.57735026, -0.57735026)
# RenderOrder -> mujoco_warp_pbr RenderOrder; PIXEL_PRIORITY (Newton's default) keeps the tuned one.
_RENDER_ORDERS = {int(RenderOrder.VIEW_PRIORITY): mjpbr.RenderOrder.ROW, int(RenderOrder.TILED): mjpbr.RenderOrder.TILED}


@wp.func
def _linear_to_srgb(value: float) -> float:
  # Newton's linear_channel_to_srgb_wp.
  clamped = wp.max(value, 0.0)
  if clamped <= 0.0031308:
    return clamped * 12.92
  return 1.055 * wp.pow(clamped, 1.0 / 2.4) - 0.055


@wp.func
def _srgb_to_linear(value: float) -> float:
  # Newton's srgb_channel_to_linear_wp.
  clamped = wp.max(value, 0.0)
  if clamped <= 0.04045:
    return clamped / 12.92
  return wp.pow((clamped + 0.055) / 1.055, 2.4)


@wp.func
def _pack_rgba(rgb: wp.vec3, alpha: float) -> wp.uint32:
  # Newton's pack_rgba_to_uint32: 0xAABBGGRR, truncated.
  r = wp.uint32(wp.clamp(rgb[0], 0.0, 1.0) * 255.0)
  g = wp.uint32(wp.clamp(rgb[1], 0.0, 1.0) * 255.0)
  b = wp.uint32(wp.clamp(rgb[2], 0.0, 1.0) * 255.0)
  a = wp.uint32(wp.clamp(alpha, 0.0, 1.0) * 255.0)
  return (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators


@wp.func
def _newton_color(c: wp.uint32, linear: bool) -> wp.uint32:
  """MuJoCo Warp's packed display colour (0xAARRGGBB, sRGB) in Newton's layout (0xAABBGGRR)."""
  if linear:
    r = _srgb_to_linear(float((c >> wp.uint32(16)) & _BYTE) / 255.0)  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
    g = _srgb_to_linear(float((c >> wp.uint32(8)) & _BYTE) / 255.0)  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
    b = _srgb_to_linear(float(c & _BYTE) / 255.0)  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
    return _pack_rgba(wp.vec3(r, g, b), 1.0)
  return (c & _ALPHA_GREEN) | ((c >> wp.uint32(16)) & _BYTE) | ((c & _BYTE) << wp.uint32(16))  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators


@wp.kernel(enable_backward=False)
def _load_rays(
  camera_rays: wp.array4d[wp.vec3],
  origin: wp.array1d[wp.vec3],
  direction: wp.array1d[wp.vec3],
  ray_scale: wp.array1d[float],
  coverage: wp.array1d[float],
):
  """Pixel rays of a Newton bundle: the sample (or mean of the samples with a direction).

  ray_scale is Newton's ray parameter per unit of distance (1 / |direction|), coverage the
  fraction of the pixel's samples that have a direction.
  """
  y, x = wp.tid()
  o = wp.vec3(0.0)
  d = wp.vec3(0.0)
  length = float(0.0)
  count = int(0)
  for s in range(camera_rays.shape[2]):
    ds = camera_rays[y, x, s, 1]
    if wp.length_sq(ds) > 0.0:
      o += camera_rays[y, x, s, 0]
      d += ds
      length += wp.length(ds)
      count += 1
  i = y * camera_rays.shape[1] + x
  if count == 0:
    origin[i] = camera_rays[y, x, 0, 0]
    direction[i] = wp.vec3(0.0)
    ray_scale[i] = 0.0
    coverage[i] = 0.0
    return
  origin[i] = o / float(count)
  direction[i] = d / float(count)
  ray_scale[i] = float(count) / length
  coverage[i] = float(count) / float(camera_rays.shape[2])


@wp.func
def _unpack_rgba(c: wp.uint32, srgb: bool) -> wp.vec4:
  """Newton's packed colour (0xAABBGGRR) as linear RGBA."""
  r = float(c & _BYTE) / 255.0  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
  g = float((c >> wp.uint32(8)) & _BYTE) / 255.0  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
  b = float((c >> wp.uint32(16)) & _BYTE) / 255.0  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
  a = float((c >> wp.uint32(24)) & _BYTE) / 255.0  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 bit operators
  if srgb:
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)
  return wp.vec4(r, g, b, a)


@wp.func
def _cover(c: wp.uint32, clear: wp.uint32, coverage: float, srgb: bool) -> wp.uint32:
  """Colour c covering a fraction of the pixel, the clear colour the rest (Newton's resolve)."""
  if coverage >= 1.0:
    return c
  mix = _unpack_rgba(c, srgb) * coverage + _unpack_rgba(clear, srgb) * (1.0 - coverage)
  rgb = wp.vec3(mix[0], mix[1], mix[2])
  if srgb:
    rgb = wp.vec3(_linear_to_srgb(rgb[0]), _linear_to_srgb(rgb[1]), _linear_to_srgb(rgb[2]))
  return _pack_rgba(rgb, mix[3])


@wp.kernel(enable_backward=False)
def _write_outputs(
  # In:
  width: int,
  view_world: wp.array1d[wp.int32],
  has_view_world: bool,
  world_count: int,
  rgb_in: wp.array2d[wp.uint32],
  depth_in: wp.array2d[float],
  seg_in: wp.array2d[wp.vec2i],
  hdr_in: wp.array2d[wp.vec3],
  albedo_in: wp.array2d[wp.vec3],
  normal_in: wp.array2d[wp.vec3],
  exposure: wp.array1d[float],
  geom_shape: wp.array2d[wp.int32],
  ray_origin: wp.array1d[wp.vec3],
  ray_direction: wp.array1d[wp.vec3],
  ray_scale: wp.array1d[float],
  coverage: wp.array1d[float],
  background: bool,
  linear: bool,
  clear_color: wp.uint32,
  clear_depth: float,
  clear_shape_index: wp.uint32,
  clear_normal: wp.vec3,
  clear_albedo: wp.uint32,
  want_color: bool,
  want_depth: bool,
  want_forward_depth: bool,
  want_shape: bool,
  want_normal: bool,
  want_albedo: bool,
  want_hdr: bool,
  # Out:
  color: wp.array3d[wp.uint32],
  depth: wp.array3d[float],
  forward_depth: wp.array3d[float],
  shape_index: wp.array3d[wp.uint32],
  normal: wp.array3d[wp.vec3],
  albedo: wp.array3d[wp.uint32],
  hdr: wp.array3d[wp.vec3],
):
  view, pixel = wp.tid()
  y = pixel / width
  x = pixel - y * width
  world = view
  if has_view_world:
    world = view_world[view]
    if world == _DISABLE_PRESERVE:
      return
  seg = seg_in[view, pixel]
  hit = seg[0] >= 0 and seg[1] == _OBJ_GEOM
  if world < 0 or world >= world_count:
    # A disabled view: cleared (its render shows world 0).
    hit = False
    background = False
    world = 0
  if not hit:
    # Newton writes the clear values where nothing is hit; a visible environment map is kept.
    if want_color:
      color[view, y, x] = wp.where(background, _newton_color(rgb_in[view, pixel], linear), clear_color)
    if want_hdr:
      hdr[view, y, x] = wp.where(background, hdr_in[view, pixel] * exposure[world % exposure.shape[0]], wp.vec3(0.0))
    if want_depth:
      depth[view, y, x] = clear_depth
    if want_forward_depth:
      forward_depth[view, y, x] = clear_depth
    if want_shape:
      shape_index[view, y, x] = clear_shape_index
    if want_normal:
      normal[view, y, x] = clear_normal
    if want_albedo:
      albedo[view, y, x] = clear_albedo
    return
  # mujoco_warp_pbr: distance along the (normalised) pixel ray.
  dist = depth_in[view, pixel]
  if want_color:
    color[view, y, x] = _cover(_newton_color(rgb_in[view, pixel], linear), clear_color, coverage[pixel], not linear)
  if want_hdr:
    hdr[view, y, x] = hdr_in[view, pixel] * exposure[world % exposure.shape[0]]
  if want_depth:
    depth[view, y, x] = dist * ray_scale[pixel]
  if want_forward_depth:
    # Along the camera's forward axis (-z), from the camera origin.
    forward_depth[view, y, x] = -(ray_origin[pixel][2] + dist * wp.normalize(ray_direction[pixel])[2])
  if want_shape:
    shape_index[view, y, x] = wp.uint32(geom_shape[world, seg[0]])
  if want_normal:
    normal[view, y, x] = normal_in[view, pixel]
  if want_albedo:
    a = albedo_in[view, pixel]
    if not linear:
      a = wp.vec3(_linear_to_srgb(a[0]), _linear_to_srgb(a[1]), _linear_to_srgb(a[2]))
    albedo[view, y, x] = _pack_rgba(a, 1.0)


def _srgb_packed_to_linear(packed: int) -> int:
  """Newton's conversion of packed sRGB clear colours for ColorSpace.LINEAR outputs."""
  r, g, b, a = (packed & 0xFF), (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF
  lr, lg, lb = (min(max(int(c * 255.0), 0), 255) for c in newton.utils.color_srgb_to_linear((r / 255.0, g / 255.0, b / 255.0)))
  return (a << 24) | (lb << 16) | (lg << 8) | lr


@dataclasses.dataclass
class _Views:
  """Render context of view_count views at one resolution and its per-frame buffers."""

  d: mjw.Data
  rc: mjpbr.PbrRenderContext[int]
  ray_scale: wp.array1d[float]
  coverage: wp.array1d[float]
  render_rgb: wp.array1d[bool]
  no_rgb: wp.array1d[bool]
  ambient: wp.array1d[wp.vec3]
  no_ambient: wp.array1d[wp.vec3]


@dataclasses.dataclass
class _Scene:
  """MuJoCo model of the Newton scene (rebuilt when the lights or materials change)."""

  mjm: mujoco.MjModel
  m: mjw.Model
  poses: ScenePoses
  views: dict[tuple, _Views]


class SensorCamera:
  """Physically based drop-in replacement for :class:`newton.sensors.SensorCamera`.

  Same constructor, :meth:`update` signature and output formats as Newton's ray-traced camera
  sensor (1.7): one view per camera transform, any of Newton's ``compute_camera_rays_*``
  bundles (pinhole, OpenCV, fisheye; camera frame -Z forward, +Y up), ``world_indices`` with
  :class:`WorldRenderFlag`, :class:`ClearData` values where nothing is hit and the
  :class:`RenderConfig` fields that have an equivalent. Switch with one import::

      from newton_pbr import SensorCamera  # instead of: from newton.sensors import SensorCamera

  Rendering follows mujoco_warp_pbr: PBR shading of the shape colours, mesh textures and scalar
  metallic / roughness, ray-traced shadows, optional image-based lighting (``envmap=``),
  path tracing (``tier=mujoco_warp_pbr.Tier.PATH``). Differences to Newton's sensor:

  * Lights: without :meth:`create_default_light`, ``lights`` or ``envmap``, the scene is lit by
    :func:`newton_pbr.default_lights` (Newton's sensor then only has ambient light).
  * Every world must have the same visible shapes (geoms are built once and posed per view).
  * Particles, deformable triangle meshes and Gaussians are not rendered; height fields neither.
  * ``RenderConfig``: ``enable_textures``, ``enable_shadows``, ``enable_ambient_lighting``,
    ``enable_backface_culling``, ``enable_fast_math``, ``output_color_space``,
    ``anti_aliasing`` and ``max_distance`` map to mujoco_warp_pbr settings; ``render_order``
    and ``block_dim`` only when changed from Newton's defaults (else mujoco_warp_pbr's tuned
    defaults); ``enable_global_world=False`` and ``TextureProjectionMode.TRIPLANAR`` are not
    supported (warned once); tile sizes, particle and Gaussian settings are ignored.
  * Multisample bundles (``sample_count > 1``, SSAA or MSAA): each pixel's ray is the mean of
    its samples and anti-aliasing uses ``ceil(sqrt(sample_count))²`` stratified samples
    interpolated between neighbouring pixel rays (mujoco_warp_pbr's supersampling). Depth,
    shape index, normal and albedo come from the first sample.
  * ``hdr_color_image`` is exposed linear radiance (before tone mapping); ``color_image`` is
    tone mapped (mujoco_warp_pbr's ``tone_mapper``) and sRGB encoded.
  * Misses show the environment map in ``color_image`` / ``hdr_color_image`` when one is set
    with ``envmap_background`` (default), else the clear colour.

  Extra keyword-only constructor options: ``lights``, ``envmap``, ``headlight`` and any
  :func:`mujoco_warp_pbr.create_render_context` option (``tier``, ``samples_per_pixel``,
  ``envmap_intensity``, ``light_model``, ``traversal``, ...), which take precedence over the
  :class:`RenderConfig` mapping.
  """

  AntiAliasing = NewtonSensorCamera.AntiAliasing
  ClearData = NewtonSensorCamera.ClearData
  GaussianRenderMode = NewtonSensorCamera.GaussianRenderMode
  RenderConfig = NewtonSensorCamera.RenderConfig
  RenderOrder = NewtonSensorCamera.RenderOrder
  TextureProjectionMode = NewtonSensorCamera.TextureProjectionMode
  Utils = NewtonSensorCamera.Utils
  WorldRenderFlag = NewtonSensorCamera.WorldRenderFlag
  Light = Light

  # Newton's ray-bundle helpers (camera-space rays, (height, width, sample_count, 2) vec3).
  compute_camera_rays_pinhole = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole)
  compute_camera_rays_pinhole_usd = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole_usd)
  compute_camera_rays_pinhole_opencv = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole_opencv)
  compute_camera_rays_fisheye_opencv = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_opencv)
  compute_camera_rays_fisheye_ftheta = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_ftheta)
  compute_camera_rays_fisheye_kannala_brandt = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_kannala_brandt)

  def __init__(
    self,
    model: newton.Model,
    *,
    default_clear_data: ClearData | None = None,
    default_render_config: RenderConfig | None = None,
    load_textures: bool = True,
    lights: list[Light] | None = None,
    envmap: Any = None,
    headlight: Headlight | None = None,
    **render_options: Any,
  ):
    """Construct a camera sensor for a model.

    Args:
      model: Newton simulation model to render; every world must have the same visible shapes.
      default_clear_data: Clear values used by :meth:`update` when its ``clear_data`` argument
        is ``None``. Defaults to ``ClearData()``.
      default_render_config: Render settings used by :meth:`update` when its
        ``render_config`` argument is ``None``. Defaults to ``RenderConfig()``.
      load_textures: Load mesh textures (shown with ``RenderConfig.enable_textures``).
      lights: Lights of the scene (Newton models carry none); None uses
        :func:`newton_pbr.default_lights` unless ``envmap`` is given or
        :meth:`create_default_light` is called.
      envmap: Equirectangular HDR environment (path or array) for image-based lighting.
      headlight: None (off) or (ambient, diffuse, specular) RGB of MuJoCo's camera headlight.
      **render_options: Further :func:`mujoco_warp_pbr.create_render_context` options; not
        those the sensor sets (nworld, cam_res, cam_active, render_rgb / depth / seg / hdr /
        guides, camera_rays).

    Raises:
      TypeError: for a render option the sensor sets itself.
      ValueError: when the worlds do not have the same visible shapes.
    """
    reserved = [name for name in _RESERVED_OPTIONS if name in render_options]
    if reserved:
      raise TypeError(f"SensorCamera sets {reserved} itself; remove them from the render options")
    self.default_clear_data: ClearData = default_clear_data if default_clear_data is not None else ClearData()
    """Clear values used by :meth:`update` when its ``clear_data`` argument is ``None``."""
    self.default_render_config: RenderConfig = default_render_config if default_render_config is not None else RenderConfig()
    """Render settings used by :meth:`update` when its ``render_config`` argument is ``None``."""
    self.model = model
    self._load_textures = load_textures
    self._lights = lights if lights is not None or envmap is not None else default_lights()
    self._headlight = headlight
    self._checkerboard: Checkerboard | None = None
    if envmap is not None:
      render_options.setdefault("envmap", envmap)
    self._render_options = render_options
    self._scene: _Scene | None = None
    self._warned: set[str] = set()
    gaussians = getattr(model, "gaussians_data", None)
    if int(model.particle_count) > 0 or (gaussians is not None and len(gaussians) > 0):
      warnings.warn(
        "newton_pbr.SensorCamera renders rigid shapes only: particles, deformable meshes and Gaussians are not rendered",
        stacklevel=2,
      )
    # Fails early for models whose worlds differ.
    self._get_scene()

  @property
  def device(self) -> wp.Device:
    """Device of the model this sensor renders."""
    return self.model.device

  def allocate_image(self, view_count: int, width: int, height: int, dtype: Any) -> wp.array[Any]:
    """Allocate a zeroed (view_count, height, width) output image array on the model device."""
    return wp.zeros((int(view_count), int(height), int(width)), dtype=dtype, device=self.device)

  def allocate_image_color(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
    """Allocate an RGBA color output array (packed ``uint32``)."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_depth(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
    """Allocate a ray-distance depth output array [m]."""
    return self.allocate_image(view_count, width, height, wp.float32)

  def allocate_image_depth_forward(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
    """Allocate a forward (planar) depth output array [m]."""
    return self.allocate_image_depth(view_count, width, height)

  def allocate_image_shape_index(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
    """Allocate a shape-index output array."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_normal(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
    """Allocate a world-space surface-normal output array (``vec3f``)."""
    return self.allocate_image(view_count, width, height, wp.vec3f)

  def allocate_image_albedo(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
    """Allocate an RGBA albedo output array (packed ``uint32``)."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_color_hdr(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
    """Allocate a linear HDR color output array (``vec3f``)."""
    return self.allocate_image(view_count, width, height, wp.vec3f)

  def compute_camera_transforms_usd(
    self,
    cameras: Any,
    *,
    time: Any | None = None,
    xform: Any | None = None,
  ) -> wp.array[wp.transformf]:
    """Read world-space camera transforms from one USD camera per view (Newton's implementation)."""
    from newton._src.core import Axis
    from newton._src.sensors.sensor_camera_render import camera_utils

    model = self.model
    return camera_utils.compute_camera_transforms_usd(
      cameras, device=model.device, target_up_axis=Axis(int(model.up_axis)), time=time, xform=xform
    )

  def create_default_light(self, *, enable_shadows: bool = True, direction: wp.vec3f | None = None) -> None:
    """Replace the lights with one directional light, as Newton's sensor does.

    Args:
      enable_shadows: Cast shadows (rendered when ``RenderConfig.enable_shadows`` is set too).
      direction: Normalized direction the light travels; defaults to normalized ``(-1, 1, -1)``.
    """
    d = _DEFAULT_LIGHT_DIRECTION if direction is None else (float(direction[0]), float(direction[1]), float(direction[2]))
    self._lights = [Light("directional", dir=d, cast_shadow=enable_shadows)]
    self._scene = None

  def assign_checkerboard_material(
    self,
    *,
    shape_indices: Sequence[int] | np.ndarray,
    resolution: int = 64,
    checker_size: int = 32,
  ) -> None:
    """Assign Newton's grey checkerboard texture to selected shapes (multiplying their colour).

    The texture is mapped with MuJoCo's texture coordinates (mesh UVs, primitive mapping), not
    Newton's cubic projection. Shown with ``RenderConfig.enable_textures``.

    Raises:
      ValueError: for a shape index out of range.
    """
    indices = np.asarray(shape_indices, dtype=np.int64).reshape(-1)
    if np.any((indices < 0) | (indices >= self.model.shape_count)):
      raise ValueError("shape_indices contains an out-of-range shape index")
    self._checkerboard = Checkerboard(frozenset(int(i) for i in indices), int(resolution), int(checker_size))
    self._scene = None

  def sync_deformable_meshes(self, state: newton.State) -> None:
    """No-op: deformable meshes are not rendered (kept for API compatibility)."""

  def _warn_once(self, key: str, message: str) -> None:
    if key not in self._warned:
      self._warned.add(key)
      warnings.warn(message, stacklevel=3)

  def _get_scene(self) -> _Scene:
    if self._scene is None:
      model = self.model
      scene = build_scene(
        model, self._lights, _FOVY, self._headlight, checkerboard=self._checkerboard, load_textures=self._load_textures
      )
      with wp.ScopedDevice(model.device):
        m = mjw.put_model(scene.mjm)
      self._scene = _Scene(scene.mjm, m, ScenePoses(model, scene), {})
    return self._scene

  def _views(self, view_count: int, width: int, height: int, config: RenderConfig, guides: bool) -> _Views:
    scene = self._get_scene()
    options = dict(self._render_options)
    options.setdefault("use_textures", bool(config.enable_textures))
    options.setdefault("enable_backface_culling", bool(config.enable_backface_culling))
    options.setdefault("use_fast_math", bool(config.enable_fast_math))
    key = (
      view_count,
      width,
      height,
      guides,
      options["use_textures"],
      options["enable_backface_culling"],
      options["use_fast_math"],
    )
    views = scene.views.get(key)
    if views is None:
      device = self.model.device
      d = put_data(scene.mjm, view_count, device)
      with wp.ScopedDevice(device):
        rc = mjpbr.create_render_context(
          scene.mjm,
          nworld=view_count,
          cam_res=(width, height),
          render_rgb=True,
          render_depth=True,
          render_seg=True,
          render_hdr=True,
          render_guides=guides,
          **options,
        )
        npix = width * height
        rays = np.zeros((1, height, width, 2, 3), dtype=np.float32)
        rays[..., 1, 2] = -1.0
        mjpbr.set_camera_rays(rc, rays)
        views = _Views(
          d=d,
          rc=rc,
          ray_scale=wp.zeros(npix, dtype=float),
          coverage=wp.zeros(npix, dtype=float),
          render_rgb=rc.base.render_rgb,
          no_rgb=wp.zeros(1, dtype=bool),
          ambient=rc.ambient_radiance,
          no_ambient=wp.zeros(1, dtype=wp.vec3),
        )
      scene.views[key] = views
    return views

  @staticmethod
  def _validate_render_array(name: str, array: Any, dtype: Any, device: wp.Device) -> None:
    if not isinstance(array, wp.array):
      raise TypeError(f"{name} must be a Warp array, got {type(array).__name__}.")
    if array.dtype != dtype:
      raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    if array.device != device:
      raise RuntimeError(f"{name} must be on the model device ({device}), got {array.device}.")

  def _apply_config(self, rc: mjpbr.PbrRenderContext[int], views: _Views, config: RenderConfig, sample_count: int) -> None:
    """Sets the per-call settings of `config` on the render context."""
    options = self._render_options
    rc.use_shadows = bool(options.get("use_shadows", config.enable_shadows))
    if "samples_per_pixel" not in options:
      rc.samples_per_pixel = max(1, math.ceil(math.sqrt(sample_count) - 1e-9))
    if "render_order" not in options and int(config.render_order) in _RENDER_ORDERS:
      rc.render_order = int(_RENDER_ORDERS[int(config.render_order)])
    if "block_dim" not in options and int(config.block_dim) != _DEFAULT_CONFIG.block_dim:
      rc.block_dim = int(config.block_dim)
    rc.base.zfar = float(config.max_distance)
    rc.ambient_radiance = views.ambient if config.enable_ambient_lighting else views.no_ambient
    if not config.enable_global_world:
      self._warn_once("global", "newton_pbr.SensorCamera: RenderConfig.enable_global_world=False is not supported")
    if int(config.texture_projection_mode) != int(TextureProjectionMode.CUBIC):
      self._warn_once("projection", "newton_pbr.SensorCamera: only MuJoCo's texture mapping is supported (no TRIPLANAR)")

  def update(
    self,
    state: newton.State,
    camera_transforms: wp.array[wp.transformf],
    camera_rays: wp.array4d[wp.vec3f],
    *,
    color_image: wp.array3d[wp.uint32] | None = None,
    depth_image: wp.array3d[wp.float32] | None = None,
    forward_depth_image: wp.array3d[wp.float32] | None = None,
    shape_index_image: wp.array3d[wp.uint32] | None = None,
    normal_image: wp.array3d[wp.vec3f] | None = None,
    albedo_image: wp.array3d[wp.uint32] | None = None,
    hdr_color_image: wp.array3d[wp.vec3f] | None = None,
    world_indices: wp.array[wp.int32] | None = None,
    clear_data: ClearData | None = None,
    render_config: RenderConfig | None = None,
    sync_deformables: bool = True,
    seed: int = 0,
  ) -> None:
    """Render this camera sensor (see :meth:`newton.sensors.SensorCamera.update`).

    The number of views is ``camera_transforms.shape[0]``; all non-``None`` outputs have shape
    ``(view_count, height, width)`` with height, width from ``camera_rays``. Unlike Newton's
    sensor, no BVH refit of the model is needed (the sensor refits its own).

    Args:
      state: Simulation state with body transforms.
      camera_transforms: World-space camera transform per view, shape ``(view_count,)``.
      camera_rays: Camera-space ray origins ``[..., 0]`` and directions ``[..., 1]``, shape
        ``(height, width, sample_count, 2)`` (Newton's ``compute_camera_rays_*``).
      color_image: Output RGBA color (packed ``uint32``, Newton's byte order).
      depth_image: Output ray-distance depth from each ray origin [m].
      forward_depth_image: Output depth along the camera's forward axis from its origin [m].
      shape_index_image: Output shape index.
      normal_image: Output world-space shading normals.
      albedo_image: Output albedo (packed ``uint32``): the surface's base colour.
      hdr_color_image: Output exposed linear HDR color (before tone mapping).
      world_indices: Optional per-view world, shape ``(view_count,)`` int32; defaults to view
        ``i`` rendering world ``i``. ``WorldRenderFlag.DISABLE_PRESERVE`` leaves a view's outputs
        unchanged; any other value outside ``[0, model.world_count)`` clears them.
      clear_data: Clear values for this call (default :attr:`default_clear_data`).
      render_config: Render settings for this call (default :attr:`default_render_config`).
      sync_deformables: Accepted for compatibility (deformables are not rendered).
      seed: Random seed of stochastic effects (soft shadows, path tracing).

    Raises:
      ValueError, TypeError, RuntimeError: for arrays of the wrong shape, dtype or device, as
        Newton's sensor.
    """
    del sync_deformables
    model = self.model
    device = model.device
    config = render_config if render_config is not None else self.default_render_config
    clear = clear_data if clear_data is not None else self.default_clear_data
    self._validate_render_array("camera_transforms", camera_transforms, wp.transformf, device)
    if camera_transforms.ndim != 1 or camera_transforms.shape[0] <= 0:
      raise ValueError(f"camera_transforms must have shape (view_count,), got {tuple(camera_transforms.shape)}.")
    self._validate_render_array("camera_rays", camera_rays, wp.vec3f, device)
    multisample_shape = camera_rays.ndim == 4 and camera_rays.shape[2] > 0 and camera_rays.shape[3] == 2
    if not multisample_shape or camera_rays.shape[0] <= 0 or camera_rays.shape[1] <= 0:
      raise ValueError(f"camera_rays must have shape (height, width, sample_count, 2), got {tuple(camera_rays.shape)}.")
    try:
      anti_aliasing = AntiAliasing(config.anti_aliasing)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Invalid anti_aliasing mode: {config.anti_aliasing!r}") from exc
    height, width, sample_count = (int(n) for n in camera_rays.shape[:3])
    if sample_count > 1 and anti_aliasing == AntiAliasing.NONE:
      raise ValueError("anti_aliasing must be SSAA or MSAA for camera_rays with more than one sample")
    view_count = int(camera_transforms.shape[0])
    world_count = int(model.world_count)
    if world_indices is None and view_count > world_count:
      raise ValueError(
        f"view_count ({view_count}) exceeds model.world_count ({world_count}); "
        "pass an explicit world_indices mapping for the extra views."
      )
    if world_indices is not None:
      if world_indices.shape != (view_count,):
        raise ValueError(f"world_indices shape must be ({view_count},), got {tuple(world_indices.shape)}")
      if world_indices.dtype != wp.int32:
        raise ValueError(f"world_indices dtype must be int32, got {world_indices.dtype}")
      if world_indices.device != device:
        raise ValueError(f"world_indices device must be {device}, got {world_indices.device}")
    images = {
      "color_image": (color_image, wp.uint32),
      "depth_image": (depth_image, wp.float32),
      "forward_depth_image": (forward_depth_image, wp.float32),
      "shape_index_image": (shape_index_image, wp.uint32),
      "normal_image": (normal_image, wp.vec3f),
      "albedo_image": (albedo_image, wp.uint32),
      "hdr_color_image": (hdr_color_image, wp.vec3f),
    }
    expected = (view_count, height, width)
    for name, (image, dtype) in images.items():
      if image is not None:
        self._validate_render_array(name, image, dtype, device)
        if tuple(image.shape) != expected:
          raise ValueError(f"{name} shape must be {expected}, got {tuple(image.shape)}")
    if all(image is None for image, _ in images.values()):
      return

    guides = normal_image is not None or albedo_image is not None
    views = self._views(view_count, width, height, config, guides)
    rc = views.rc
    shaded = guides or color_image is not None or hdr_color_image is not None
    rc.base.render_rgb = views.render_rgb if shaded else views.no_rgb
    self._apply_config(rc, views, config, sample_count)
    if rc.ray_origin is None or rc.ray_direction is None:
      raise RuntimeError("newton_pbr.SensorCamera: the render context lost its camera rays")
    outputs = [rc.ray_origin, rc.ray_direction, views.ray_scale, views.coverage]
    wp.launch(_load_rays, dim=(height, width), inputs=[camera_rays], outputs=outputs)
    scene = self._get_scene()
    scene.poses.sync(state, camera_transforms, views.d, view_world=world_indices)
    if rc.traversal != "optix":  # OptiX updates its own structures in render
      mjw.refit_bvh(scene.m, views.d, rc.base)
    mjpbr.render(scene.m, views.d, rc, seed=seed)

    linear = int(config.output_color_space) == int(ColorSpace.LINEAR)
    clear_color, clear_albedo = int(clear.clear_color), int(clear.clear_albedo)
    if linear:
      clear_color, clear_albedo = _srgb_packed_to_linear(clear_color), _srgb_packed_to_linear(clear_albedo)
    base = rc.base
    dummy_u = self._placeholder(wp.uint32)
    dummy_f = self._placeholder(wp.float32)
    dummy_v = self._placeholder(wp.vec3f)
    wp.launch(
      _write_outputs,
      dim=(view_count, width * height),
      inputs=[
        width,
        world_indices if world_indices is not None else scene.poses._no_world,
        world_indices is not None,
        world_count,
        base.rgb_data,
        base.depth_data,
        base.seg_data,
        rc.hdr_data,
        rc.guide_albedo,
        rc.guide_normal,
        rc.exposure,
        scene.poses.geom_shape,
        rc.ray_origin,
        rc.ray_direction,
        views.ray_scale,
        views.coverage,
        bool(rc.env_background),
        linear,
        wp.uint32(clear_color),
        float(clear.clear_depth),
        wp.uint32(int(clear.clear_shape_index)),
        wp.vec3(*clear.clear_normal),
        wp.uint32(clear_albedo),
        color_image is not None,
        depth_image is not None,
        forward_depth_image is not None,
        shape_index_image is not None,
        normal_image is not None,
        albedo_image is not None,
        hdr_color_image is not None,
      ],
      outputs=[
        color_image if color_image is not None else dummy_u,
        depth_image if depth_image is not None else dummy_f,
        forward_depth_image if forward_depth_image is not None else dummy_f,
        shape_index_image if shape_index_image is not None else dummy_u,
        normal_image if normal_image is not None else dummy_v,
        albedo_image if albedo_image is not None else dummy_u,
        hdr_color_image if hdr_color_image is not None else dummy_v,
      ],
      device=device,
    )

  def _placeholder(self, dtype: Any) -> wp.array3d[Any]:
    """A (1, 1, 1) array bound to the outputs update() is not asked for."""
    cache = self.__dict__.setdefault("_placeholders", {})
    if dtype not in cache:
      cache[dtype] = wp.empty((1, 1, 1), dtype=dtype, device=self.device)
    return cache[dtype]
