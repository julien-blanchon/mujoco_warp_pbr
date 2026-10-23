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

import dataclasses
import enum
import math
import os
import warnings
from typing import Optional, Sequence, TypeVar, Union

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import bvh as mjw_bvh

from mujoco_warp_pbr._src import color
from mujoco_warp_pbr._src import envmap as envmap_lib
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import pathtrace
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src import traverse
from mujoco_warp_pbr._src.types import LightModel
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder
from mujoco_warp_pbr._src.types import Tier
from mujoco_warp_pbr._src.types import ToneMapper
from mujoco_warp_pbr._src.types import UnsetMaterial

# Filament's default camera: f/16, 1/125 s, ISO 100.
DEFAULT_EV100 = 14.96578428466209
# Register cap of the render kernels when max_registers is None (tuned on Hopper).
DEFAULT_MAX_REGISTERS = 96
# Threads per block for primitive-only scenes when block_dim is None (Ampere / Hopper).
PRIMITIVE_BLOCK_DIM = 256
# First compute capability of Blackwell RTX GPUs, whose tuned defaults differ.
_BLACKWELL_RTX_ARCH = 120
# Re-frame a mesh BVH when its summed triangle bounding-box area shrinks by at least this factor.
_MESH_FRAME_MIN_GAIN = 1.25

_TRAVERSALS = ("software", "optix", "auto")
_MESH_BVH_FRAMES = ("auto", "mujoco")
_MESH_BVH_CONSTRUCTORS = ("cubql", "sah", "lbvh")
_MATERIAL_KEYS = ("metallic", "roughness")

EnvmapItem = Union[str, os.PathLike, np.ndarray]
_Enum = TypeVar("_Enum", bound=enum.IntEnum)


def _enum(cls: type[_Enum], value: int, name: str) -> _Enum:
  try:
    return cls(value)
  except ValueError:
    options = ", ".join(f"{cls.__name__}.{e.name}" for e in cls)
    raise ValueError(f"{name} must be one of {options}, got {value!r}") from None


def _check_at_least(name: str, value: int, low: int):
  if value < low:
    raise ValueError(f"{name} must be at least {low}, got {value}")


def _check_default_material(default_material: Optional[dict]):
  for key, value in (default_material or {}).items():
    if key not in _MATERIAL_KEYS:
      raise ValueError(f"default_material keys must be among {_MATERIAL_KEYS}, got {key!r}")
    if not 0.0 <= value <= 1.0:
      raise ValueError(f"default_material[{key!r}] must be in [0, 1], got {value}")


def _warn_unsupported(mjm: mujoco.MjModel):
  if mjm.nflex and np.any(mjm.mat_texid[:, mujoco.mjtTextureRole.mjTEXROLE_NORMAL] >= 0):
    warnings.warn("flex geometry is rendered without normal maps", stacklevel=4)
  image_lights = np.nonzero(mjm.light_type == lights.LIGHT_IMAGE)[0]
  if image_lights.size:
    warnings.warn(f"image lights are not supported and are ignored: {image_lights.tolist()}", stacklevel=4)


def _select_traversal(
  mjm: mujoco.MjModel, enabled_geom_groups: Sequence[int], tier: Tier, traversal: str, rendered: np.ndarray
) -> str:
  """Resolves traversal="auto" and falls back from "optix" when it cannot render the model."""
  if traversal == "software":
    return traversal
  from mujoco_warp_pbr._src import optix as optix_backend

  if traversal == "auto":
    has_mesh = bool(np.any(mjm.geom_type[rendered] == mujoco.mjtGeom.mjGEOM_MESH))
    if not has_mesh:
      return "software"
    return "optix" if optix_backend.unavailable_reason(mjm, enabled_geom_groups, tier) is None else "software"
  reason = optix_backend.unavailable_reason(mjm, enabled_geom_groups, tier)
  if reason is not None:
    warnings.warn(f'traversal="optix" falls back to "software": {reason}', stacklevel=3)
    return "software"
  return traversal


def _bbox_area_sum(points: np.ndarray, faces: np.ndarray) -> float:
  """Summed surface area of the triangles' axis-aligned bounding boxes (a BVH cost proxy)."""
  tri = points[faces]
  e = tri.max(axis=1) - tri.min(axis=1)
  return float(np.sum(e[:, 0] * e[:, 1] + e[:, 1] * e[:, 2] + e[:, 2] * e[:, 0]))


def _mesh_frames(mjm: mujoco.MjModel) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Authored-frame rotations of the meshes whose BVH they improve: {mesh_id: (R, R v, faces)}.

  MuJoCo's mesh frame is the principal inertia frame; R (from mesh_quat) rotates its vertices v
  back to the authored frame. A mesh is kept when that shrinks its summed triangle bounding-box
  area by at least _MESH_FRAME_MIN_GAIN.
  """
  out = {}
  for mesh_id in range(mjm.nmesh):
    if np.allclose(mjm.mesh_quat[mesh_id], [1.0, 0.0, 0.0, 0.0]):
      continue
    v0, nv = mjm.mesh_vertadr[mesh_id], mjm.mesh_vertnum[mesh_id]
    f0, nf = mjm.mesh_faceadr[mesh_id], mjm.mesh_facenum[mesh_id]
    points = mjm.mesh_vert[v0 : v0 + nv].astype(np.float64)
    faces = mjm.mesh_face[f0 : f0 + nf]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, mjm.mesh_quat[mesh_id])
    # mesh_quat rotates the compiled vertices back to the authored frame.
    rot = rot.reshape(3, 3)
    framed = points @ rot.T
    if _bbox_area_sum(points, faces) < _MESH_FRAME_MIN_GAIN * _bbox_area_sum(framed, faces):
      continue
    out[mesh_id] = (rot, framed, faces)
  return out


def _rebuild_mesh_bvhs(mjm: mujoco.MjModel, base: mjw.RenderContext, constructor: str, skip: set[int]):
  """Rebuilds the mesh BVHs of `base` (built with cuBQL by MuJoCo Warp) with another builder.

  The replaced BVHs are dropped from `base.mesh_registry`. Meshes in `skip` keep theirs: they are
  traced through re-framed copies (see `_mesh_bvh_frames`).
  """
  ids = base.mesh_bvh_id.numpy().copy()  # numpy() is a view of CPU arrays
  for mesh_id in range(mjm.nmesh):
    if mesh_id in skip:
      continue
    mesh, _ = mjw_bvh.build_mesh_bvh(mjm, mesh_id, constructor=constructor)
    base.mesh_registry.pop(int(ids[mesh_id]), None)
    base.mesh_registry[mesh.id] = mesh
    ids[mesh_id] = mesh.id
  if mjm.nmesh:
    base.mesh_bvh_id = wp.array(ids, dtype=wp.uint64)


def _mesh_bvh_frames(
  mjm: mujoco.MjModel, base: mjw.RenderContext, nworld: int, selected: dict, constructor: str, traversal: str
) -> dict:
  """PbrRenderContext fields of the meshes traced in their authored frame (see trace.py).

  The software traversal gets re-framed Warp mesh BVHs; the OptiX backend only the rotations (it
  builds its triangle GASes from them, see optix/accel.py).
  """
  frames = np.tile(np.eye(3, dtype=np.float32), (mjm.nmesh, 1, 1))
  for mesh_id, (rot, _, _) in selected.items():
    frames[mesh_id] = rot
  if traversal != "software":
    return {"mesh_bvh_frame": wp.array(frames, dtype=wp.mat33)}
  ids = base.mesh_bvh_id.numpy().copy()  # numpy() is a view of CPU arrays
  registry = []
  for mesh_id, (_, framed, faces) in selected.items():
    mesh = wp.Mesh(
      points=wp.array(framed.astype(np.float32), dtype=wp.vec3),
      indices=wp.array(faces.flatten().astype(np.int32), dtype=wp.int32),
      bvh_constructor=constructor,
      bvh_leaf_size=2,
    )
    registry.append(mesh)
    ids[mesh_id] = mesh.id
  return {
    "mesh_bvh_frame": wp.array(frames, dtype=wp.mat33),
    "trace_mesh_bvh_id": wp.array(ids, dtype=wp.uint64),
    "trace_mesh_registry": registry,
    "geom_xmat_trace": wp.zeros((nworld, mjm.ngeom), dtype=wp.mat33),
  }


@dataclasses.dataclass
class _Lighting:
  exposure: float
  light_intensity: np.ndarray
  # Headlight illuminance times exposure (display-relative, see PbrRenderContext).
  headlight_intensity: float
  ambient: np.ndarray


def _resolve_lighting(mjm: mujoco.MjModel, light_model: LightModel, ev100: float) -> _Lighting:
  physical = light_model == LightModel.PHYSICAL
  exposure = color.exposure_from_ev100(ev100) if physical else 1.0
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  # The kernel scales the light's diffuse colour by this intensity. Authored intensities carry
  # the brightness, so the colour is normalised; unset ones keep MuJoCo's colour brightness.
  light_intensity = np.array(mjm.light_intensity, dtype=np.float32)
  for i in range(mjm.nlight):
    if light_intensity[i] > 0.0:
      light_intensity[i] /= max(float(np.max(mjm.light_diffuse[i])), 1e-6)
    else:
      light_intensity[i] = lights.fallback_intensity(mjm, mjd, i, exposure)

  # Constant ambient radiance until an environment is supplied: in the physical model the
  # headlight ambient is display-relative (scaled by 1 / exposure); the MuJoCo model also adds
  # each light's ambient colour.
  ambient = np.zeros(3, dtype=np.float32)
  if mjm.vis.headlight.active:
    ambient += mjm.vis.headlight.ambient
  if physical:
    ambient /= exposure
  else:
    ambient += np.sum(mjm.light_ambient[mjm.light_active.astype(bool)], axis=0)
  return _Lighting(exposure, light_intensity, math.pi, ambient)


def _envmap_images(envmap: Union[EnvmapItem, Sequence[EnvmapItem], None]) -> list[np.ndarray]:
  if envmap is None:
    return []
  items = envmap if isinstance(envmap, (list, tuple)) else [envmap]
  images = []
  for i, item in enumerate(items):
    if isinstance(item, (str, bytes, os.PathLike)):
      images.append(envmap_lib.load_hdr(os.fsdecode(item)))
      continue
    image = np.asarray(item)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.shape[0] < 1 or image.shape[1] < 1:
      raise ValueError(f"envmap[{i}] must be a path or a (height, width, 3) radiance array, got shape {image.shape}")
    images.append(image)
  return images


def _build_environment(
  mjm: mujoco.MjModel, envmap: Union[EnvmapItem, Sequence[EnvmapItem], None], use_skybox: bool, physical: bool
) -> tuple[envmap_lib.Environments, bool, bool]:
  """Environment maps: explicit ones, else the MJCF skybox, else constant ambient.

  Returns (environments, whether they light the scene, whether there is any map).
  """
  images = _envmap_images(envmap)
  skybox_only = False
  if not images and use_skybox:
    sky = envmap_lib.skybox_image(mjm)
    if sky is not None:
      images.append(sky)
      # MuJoCo's GL lights never come from the skybox: there it is background only.
      skybox_only = not physical
  if not images:
    return envmap_lib.empty_environments(), False, False
  # Looked up at call time: benchmarks share built environments between contexts by patching it.
  return envmap_lib.build_environments(images), not skybox_only, True


def _surface_flags(
  mjm: mujoco.MjModel, rendered: np.ndarray, rendered_mats: np.ndarray, features: material.SurfaceFeatures, reflections: bool
) -> tuple[bool, bool]:
  """(has_transparency, has_reflection) of the rendered geoms."""
  alpha = material.effective_geom_rgba(mjm)[rendered, 3]
  has_transparency = bool(np.any((alpha > 0.0) & (alpha < 1.0))) or features.alpha_map
  has_transparency = has_transparency or bool(mjm.nflex and np.any(mjm.flex_rgba[:, 3] < 1.0))
  has_reflection = reflections and bool(np.any(mjm.mat_reflectance[rendered_mats] > 0.0))
  return has_transparency, has_reflection


def _tuning_defaults(
  base: mjw.RenderContext,
  tier: Tier,
  block_dim: Optional[int],
  max_registers: Optional[int],
  ordered_traversal: Optional[bool],
) -> tuple[Optional[int], int, bool]:
  """Measured defaults of block_dim, max_registers and ordered_traversal (results/hf/tune_*.jsonl).

  On Ampere / Hopper, primitive-only scenes run fastest with 256-thread blocks and the ordered
  traversal; on Blackwell RTX (sm_120) with 64-thread blocks and MuJoCo Warp's traversal (both
  ~25% faster there).
  """
  mesh_like = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
  has_meshes = bool(mesh_like & set(base.geom_ray_types)) or base.bvh_nflexgeom > 0
  device = wp.get_device()
  blackwell_rtx = device.is_cuda and device.arch >= _BLACKWELL_RTX_ARCH
  if block_dim is None and not (has_meshes or blackwell_rtx):
    block_dim = PRIMITIVE_BLOCK_DIM
  if max_registers is None:
    max_registers = DEFAULT_MAX_REGISTERS
  if ordered_traversal is None:
    ordered_traversal = tier == Tier.REALTIME and bool(base.render_rgb.numpy().any())
    ordered_traversal = ordered_traversal and (has_meshes or not blackwell_rtx)
  ordered_traversal = ordered_traversal and traverse.AVAILABLE and base.bvh_nflexgeom == 0
  return block_dim, max_registers, ordered_traversal


def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int = 1,
  cam_res: list[tuple[int, int]] | tuple[int, int] | None = None,
  render_rgb: list[bool] | bool | None = None,
  render_depth: list[bool] | bool | None = None,
  render_seg: list[bool] | bool | None = None,
  enabled_geom_groups: Sequence[int] = (0, 1, 2),
  cam_active: list[bool] | list[str] | list[int] | None = None,
  background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
  use_fast_math: bool = True,
  use_shadows: bool = True,
  shadow_light_fraction: float = 0.0,
  enable_backface_culling: bool = True,
  enable_vertex_normals: bool = True,
  light_model: LightModel | int = LightModel.PHYSICAL,
  tone_mapper: ToneMapper | int = ToneMapper.PBR_NEUTRAL,
  ev100: float = DEFAULT_EV100,
  unset_material: UnsetMaterial | int = UnsetMaterial.GLTF,
  default_material: Optional[dict[str, float]] = None,
  use_textures: bool = True,
  samples_per_pixel: int = 1,
  specular_aa: bool = True,
  shadow_samples: int = 1,
  max_layers: int = 4,
  enable_reflections: bool = True,
  envmap: Union[EnvmapItem, Sequence[EnvmapItem], None] = None,
  envmap_intensity: float | None = None,
  envmap_rotation: float = 0.0,
  envmap_background: bool = True,
  use_skybox: bool = True,
  render_order: RenderOrder | int = RenderOrder.TILED,
  block_dim: int | None = None,
  max_registers: int | None = None,
  mesh_bvh_constructor: str = "cubql",
  mesh_bvh_frame: str = "auto",
  ordered_traversal: bool | None = None,
  tier: Tier | int = Tier.REALTIME,
  path_samples: int = 64,
  max_depth: int = 4,
  rr_depth: int = 3,
  denoise: bool = False,
  denoise_iterations: int = 3,
  render_hdr: bool = False,
  traversal: str = "software",
) -> PbrRenderContext:
  """Creates a PBR render context on the current Warp device.

  Camera selection, resolutions, geom groups and the rgb / depth / segmentation switches
  follow `mujoco_warp.create_render_context`. Every array is allocated on Warp's current device:
  create the context (and the MuJoCo Warp model and data) inside `with wp.ScopedDevice(device):`
  to render on another device, e.g. "cpu" or "cuda:1".

  Args:
    mjm: The model on host.
    nworld: The number of worlds.
    cam_res: Width and height per active camera; None uses the model values.
    render_rgb: Whether to render RGB per camera; None uses the model values.
    render_depth: Whether to render depth per camera; None uses the model values.
    render_seg: Whether to render segmentation per camera; None uses the model values.
    enabled_geom_groups: Geom groups to render.
    cam_active: Cameras to render as booleans, names or indices; None renders all.
    background_color: Display (sRGB) colour of pixels that hit nothing.
    use_fast_math: Compile the render kernel with fast math.
    use_shadows: Trace shadow rays for lights with castshadow.
    shadow_light_fraction: Fraction of direct light reaching occluded points, in [0, 1].
    enable_backface_culling: Drop hits whose normal faces away from the ray (ray starts inside).
    enable_vertex_normals: Shade meshes with interpolated vertex normals.
    light_model: LightModel.PHYSICAL or LightModel.MUJOCO.
    tone_mapper: ToneMapper applied before sRGB encoding.
    ev100: Camera exposure value for the physical light model.
    unset_material: UnsetMaterial policy for metallic / roughness left unset (-1).
    default_material: Optional {"metallic": .., "roughness": ..} (values in [0, 1]) applied to
      unset fields only.
    use_textures: Sample material textures (all roles).
    samples_per_pixel: Stratified sub-pixel samples per axis; N gives N×N rays per pixel.
    specular_aa: Widen roughness by the normal variance across the pixel footprint.
    shadow_samples: Shadow rays per light; > 1 gives soft shadows from the light's bulbradius.
    max_layers: Translucent surfaces composited front to back along a camera ray.
    enable_reflections: Trace one mirror ray for materials with MuJoCo reflectance > 0.
    envmap: Environment lighting: a path to an equirectangular .hdr file, a float array
      (height, width, 3) of linear radiance, or a list of those (select per world with `env_id`).
      Row 0 is the zenith (+z); the centre column looks along +x.
    envmap_intensity: Radiance scale of the environment map. None makes it display-relative
      (a texel of 1 shows as white at the default exposure).
    envmap_rotation: Rotation of the environment about +z in degrees.
    envmap_background: Show the environment map where rays hit nothing.
    use_skybox: Without `envmap`, light the scene (physical light model) and fill the
      background with the model's skybox texture, if it has one.
    render_order: RenderOrder assignment of pixels to threads (see RenderOrder).
    block_dim: Threads per block of the render kernels. None picks 256 for primitive-only scenes
      and MuJoCo Warp's render block size (64) when meshes, heightfields or flex are rendered
      (mesh traversal needs more registers and local memory per thread), and 64 on Blackwell
      RTX GPUs (sm_120), where it is faster for every scene.
    max_registers: Register cap per thread for the render kernels. None picks 96, 5-12% faster
      than the compiler's choice on Hopper for both primitive and mesh scenes; 0 leaves the
      register count to the compiler.
    ordered_traversal: Traverse the scene BVH nearest child first (`traverse.py`; needs
      Warp >= 1.18; scenes with flex always use MuJoCo Warp's traversal). None picks it for the
      real-time tier with RGB output, where it is up to 24% faster (shading kernels trace
      incoherent shadow and mirror rays), except for primitive-only scenes on Blackwell RTX GPUs
      (sm_120), where it is slower; depth / segmentation-only and path-traced kernels are faster
      with MuJoCo Warp's traversal.
    mesh_bvh_constructor: Warp BVH builder for meshes ("cubql" as MuJoCo Warp, "sah", "lbvh").
    mesh_bvh_frame: Frame mesh BVHs (software traversal) and triangle GASes (OptiX) are built in.
      "mujoco" uses MuJoCo's mesh frame (the principal inertia frame) as MuJoCo Warp does. "auto"
      (default) rebuilds a mesh's BVH in its authored frame when that shrinks its triangles'
      bounding boxes substantially: CAD-style meshes are axis-aligned as authored, and rotated
      into the inertia frame their long triangles get large overlapping boxes (up to ~5x slower
      rendering).
    tier: Tier.REALTIME (direct lighting + split-sum IBL) or Tier.PATH (path tracing).
    path_samples: Path-traced samples per pixel (random sub-pixel positions).
    max_depth: Maximum path length in bounces (path-traced tier).
    rr_depth: Bounce after which Russian roulette may terminate paths (path-traced tier).
    denoise: Filter the path-traced image with an edge-avoiding à-trous denoiser guided by
      first-hit albedo, normal and depth (Tier.PATH only; off by default).
    denoise_iterations: Filter passes (footprint 4 · 2^(iterations - 1) + 1 pixels).
    render_hdr: Also write linear HDR radiance (before exposure and tone mapping) to `hdr_data`.
    traversal: Ray traversal of the real-time tier. "software" (default) traverses MuJoCo Warp's
      scene BVH in Warp. "optix" traces through OptiX acceleration structures, on RT cores where
      the GPU has them (needs PyOptiX and the OptiX SDK headers, see `_src/optix`); it falls back
      to "software" with a warning when unavailable or when the scene has flex, height fields or
      SDF geoms. Depth, segmentation and images match the software traversal up to rounding.
      "auto" picks "optix" when it is available and meshes are rendered (3-7x faster than
      software on the mesh scenes measured on RTX PRO 6000), else "software": on
      primitive-only scenes OptiX ranged from 0.6x (mirror reflections) to 1.3x of software.

  Returns:
    The PBR render context.

  Raises:
    ValueError: for invalid arguments, or a model with nothing to render.
  """
  light_model = _enum(LightModel, light_model, "light_model")
  tone_mapper = _enum(ToneMapper, tone_mapper, "tone_mapper")
  unset_material = _enum(UnsetMaterial, unset_material, "unset_material")
  render_order = _enum(RenderOrder, render_order, "render_order")
  tier = _enum(Tier, tier, "tier")
  for name, value in (
    ("samples_per_pixel", samples_per_pixel),
    ("shadow_samples", shadow_samples),
    ("max_layers", max_layers),
    ("path_samples", path_samples),
  ):
    _check_at_least(name, value, 1)
  for name, value in (("max_depth", max_depth), ("rr_depth", rr_depth), ("denoise_iterations", denoise_iterations)):
    _check_at_least(name, value, 0)
  if tier == Tier.PATH and max_depth + 4 * max_layers >= pathtrace.MAX_PATH_VERTICES:
    raise ValueError(
      f"max_depth + 4 * max_layers must be below {pathtrace.MAX_PATH_VERTICES} for the path tracer's random "
      f"number layout, got {max_depth} + 4 * {max_layers}"
    )
  if not 0.0 <= shadow_light_fraction <= 1.0:
    raise ValueError(f"shadow_light_fraction must be in [0, 1], got {shadow_light_fraction}")
  _check_default_material(default_material)
  if denoise and tier != Tier.PATH:
    raise ValueError("denoise applies to the path-traced tier (Tier.PATH)")
  if mesh_bvh_frame not in _MESH_BVH_FRAMES:
    raise ValueError(f'mesh_bvh_frame must be "auto" or "mujoco", got {mesh_bvh_frame!r}')
  if mesh_bvh_constructor not in _MESH_BVH_CONSTRUCTORS:
    raise ValueError(f"mesh_bvh_constructor must be one of {_MESH_BVH_CONSTRUCTORS}, got {mesh_bvh_constructor!r}")
  if traversal not in _TRAVERSALS:
    raise ValueError(f'traversal must be "software", "optix" or "auto", got {traversal!r}')
  if block_dim is not None:
    _check_at_least("block_dim", block_dim, 1)
  if max_registers is not None:
    _check_at_least("max_registers", max_registers, 0)

  _warn_unsupported(mjm)
  rendered = np.isin(mjm.geom_group, list(enabled_geom_groups))
  if not np.any(rendered) and not mjm.nflex:
    raise ValueError("the model has no geoms in the enabled geom groups to render")
  traversal = _select_traversal(mjm, enabled_geom_groups, tier, traversal, rendered)
  base = mjw.create_render_context(
    mjm,
    nworld=nworld,
    cam_res=cam_res,
    render_rgb=render_rgb,
    render_depth=render_depth,
    render_seg=render_seg,
    use_textures=False,
    use_fast_math=use_fast_math,
    use_shadows=use_shadows,
    enabled_geom_groups=list(enabled_geom_groups),
    cam_active=cam_active,
    background_color=background_color,
    use_precomputed_rays=False,
    render_skybox=False,
    enable_backface_culling=enable_backface_culling,
    enable_vertex_normals=enable_vertex_normals,
  )

  reframed = _mesh_frames(mjm) if mesh_bvh_frame == "auto" else {}
  if mesh_bvh_constructor != "cubql":
    # Meshes re-framed for the software traversal are traced through their re-framed BVH only.
    _rebuild_mesh_bvhs(mjm, base, mesh_bvh_constructor, set(reframed) if traversal == "software" else set())
  frames = _mesh_bvh_frames(mjm, base, nworld, reframed, mesh_bvh_constructor, traversal) if reframed else {}

  physical = light_model == LightModel.PHYSICAL
  lighting = _resolve_lighting(mjm, light_model, ev100)
  metallic, roughness, reflectance = material.resolve_geom_materials(mjm, unset_material, default_material)

  # Only materials of rendered geoms decide which features the kernel uses.
  rendered_mats = np.unique(mjm.geom_matid[rendered & (mjm.geom_matid >= 0)])
  # The path tracer resolves highlights and texture detail by sampling: no footprint filtering.
  features = material.SurfaceFeatures(mjm, use_textures, specular_aa and tier == Tier.REALTIME, rendered_mats)
  used_tex = np.zeros(mjm.ntex, dtype=bool)
  if features.textures:
    ids = mjm.mat_texid[rendered_mats].ravel()
    used_tex[ids[ids >= 0]] = True
  textures, tex_size = texture.upload_textures(mjm, used_tex)
  has_transparency, has_reflection = _surface_flags(mjm, rendered, rendered_mats, features, enable_reflections)
  background_linear = color.srgb_to_linear(background_color[:3])

  env, has_env, any_env = _build_environment(mjm, envmap, use_skybox, physical)
  if envmap_intensity is None:
    envmap_intensity = 1.0 / lighting.exposure
  if has_env and physical:
    # The environment replaces MuJoCo's ambient terms.
    lighting.ambient[:] = 0.0

  block_dim, max_registers, ordered_traversal = _tuning_defaults(base, tier, block_dim, max_registers, ordered_traversal)

  npixel = base.rgb_data.shape[1]
  aov_pixels = npixel if denoise else 1

  def per_world(values: np.ndarray) -> wp.array:
    """A (1, n) batched float array: shared by every world until replaced."""
    return wp.array(np.asarray(values, dtype=np.float32).reshape(1, -1), dtype=float)

  return PbrRenderContext(
    base=base,
    light_model=int(light_model),
    tone_mapper=int(tone_mapper),
    use_shadows=use_shadows,
    shadow_light_fraction=shadow_light_fraction,
    geom_metallic=per_world(metallic),
    geom_roughness=per_world(roughness),
    geom_reflectance=per_world(reflectance),
    mat_reflectance=per_world(mjm.mat_reflectance),
    light_intensity=per_world(lighting.light_intensity),
    light_range=per_world(mjm.light_range),
    light_bulbradius=per_world(mjm.light_bulbradius),
    headlight_intensity=lighting.headlight_intensity,
    exposure=wp.array([lighting.exposure], dtype=float),
    ambient_radiance=wp.array([lighting.ambient], dtype=wp.vec3),
    dfg_lut=ibl.build_dfg_lut(),
    background_linear=wp.vec3(*background_linear),
    has_env=has_env,
    env_background=any_env and envmap_background,
    env_maps=env.maps,
    env_radiance=env.radiance,
    env_prefiltered=env.prefiltered,
    env_sh=env.sh,
    env_texel_angle=envmap_lib.texel_angle(env.width),
    env_prob=env.prob,
    env_marginal=env.marginal,
    env_conditional=env.conditional,
    env_id=wp.zeros(1, dtype=int),
    env_intensity=wp.array([envmap_intensity], dtype=float),
    env_yaw=wp.array([np.radians(envmap_rotation)], dtype=float),
    textures=wp.array(textures, dtype=wp.Texture2D),
    textures_registry=textures,
    tex_size=wp.array(tex_size, dtype=wp.vec2),
    features=features,
    samples_per_pixel=samples_per_pixel,
    shadow_samples=shadow_samples,
    has_transparency=has_transparency,
    max_layers=max_layers,
    has_reflection=has_reflection,
    render_order=int(render_order),
    block_dim=block_dim,
    max_registers=max_registers,
    ordered_traversal=ordered_traversal,
    tier=int(tier),
    path_samples=path_samples,
    max_depth=max_depth,
    rr_depth=rr_depth,
    denoise=denoise,
    denoise_iterations=denoise_iterations,
    aov_albedo=wp.zeros((nworld, aov_pixels), dtype=wp.vec3h),
    aov_normal=wp.zeros((nworld, aov_pixels), dtype=wp.vec3h),
    aov_depth=wp.zeros((nworld, aov_pixels), dtype=float),
    aov_emission=wp.zeros((nworld, aov_pixels), dtype=wp.vec3),
    aov_variance=wp.zeros((nworld, aov_pixels), dtype=float),
    denoise_buffers=(wp.zeros((nworld, aov_pixels), dtype=wp.vec4), wp.zeros((nworld, aov_pixels), dtype=wp.vec4)),
    render_hdr=render_hdr,
    hdr_data=wp.zeros((nworld, npixel if render_hdr or denoise else 1), dtype=wp.vec3),
    traversal=traversal,
    **frames,
  )
