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
from typing import TYPE_CHECKING, Any, Optional

import warp as wp
from mujoco_warp import RenderContext

if TYPE_CHECKING:
  from mujoco_warp_pbr._src.material import SurfaceFeatures


class LightModel(enum.IntEnum):
  """How MuJoCo light parameters are interpreted.

  Attributes:
    PHYSICAL: `light.intensity` in lux (directional) or candela (point, spot), inverse-square
      falloff windowed at `light.range`, smooth spot cone, EV100 exposure. Lights with intensity 0
      use a calibrated fallback so default MuJoCo scenes are reasonably exposed.
    MUJOCO: MuJoCo's OpenGL semantics (attenuation polynomial, spot exponent, diffuse /
      specular / ambient colours) feeding the PBR BRDF, exposure 1.
  """

  PHYSICAL = 0
  MUJOCO = 1


class Tier(enum.IntEnum):
  """Rendering algorithm.

  Attributes:
    REALTIME: primary ray, ray-traced shadows, split-sum image-based lighting, optional mirror
      ray and translucent layers. Deterministic and noise-free.
    PATH: unidirectional path tracing with next-event estimation and multiple importance
      sampling of lights, environment and BSDF. Converges to the reference image.
  """

  REALTIME = 0
  PATH = 1


class ToneMapper(enum.IntEnum):
  """Display transform applied to exposed linear HDR before sRGB encoding.

  Attributes:
    PBR_NEUTRAL: Khronos PBR Neutral, preserves base colours under neutral lighting.
    ACES: ACES RRT + ODT fit.
    CLAMP: no tone mapping, clamp to [0, 1].
  """

  PBR_NEUTRAL = 0
  ACES = 1
  CLAMP = 2


class RenderOrder(enum.IntEnum):
  """Assignment of pixels to threads (outputs are always stored row-major).

  Attributes:
    ROW: thread order follows image rows; a warp covers 32 pixels of one row.
    TILED: 8-row bands traversed column by column; a warp covers a 4 × 8 pixel tile. Default:
      ~15-20% faster than ROW whether worlds look alike or not.
    WORLD: neighbouring threads render the same pixel in consecutive worlds. Up to ~2.7× faster
      when worlds render nearly identical images (fixed cameras, small state differences), but
      ~2× slower than TILED when they differ a lot (e.g. randomised camera poses).
  """

  ROW = 0
  TILED = 1
  WORLD = 2


class UnsetMaterial(enum.IntEnum):
  """How material fields MuJoCo leaves unset (metallic / roughness = -1) are resolved.

  Attributes:
    GLTF: metallic 0, roughness 1 - shininess.
    HEURISTIC: colour-based guesses in the style of mujofil (off by default).
  """

  GLTF = 0
  HEURISTIC = 1


# Settings compiled into the kernels by default (bake groups of `kernel_inputs.BAKE_GROUPS`).
DEFAULT_BAKED = frozenset(
  {
    "render_order",
    "total_rays",
    "samples_per_axis",
    "clip",
    "tone_mapper",
    "render_hdr",
    "any_rgb",
    "light_model",
    "has_spot",
    "use_shadows",
    "shadow_samples",
    "shadow_light_fraction",
    "headlight_active",
    "headlight",
    "has_env",
    "env_background",
    "inv_texel_angle",
    "textures",
    "specular_aa",
    "vertex_normals",
    "has_reflection",
    "transparency",
    "denoise",
    "nlight",
  }
)


@dataclasses.dataclass
class PbrRenderContext:
  """Context for physically based rendering.

  Geometry (BVHs, cameras, rays) and the rgb / depth / segmentation output buffers live in
  `base`, a regular MuJoCo Warp render context, so `mujoco_warp.get_rgb` / `get_depth` /
  `get_segmentation` work on `base` unchanged.

  Arrays whose first dimension is a batch dimension are indexed with `worldid % shape[0]`,
  following MuJoCo Warp's convention: replace them with arrays of leading size nworld to vary
  a value per world.

  Attributes:
    base: MuJoCo Warp render context holding geometry and outputs.
    light_model: LightModel.
    tone_mapper: ToneMapper.
    use_shadows: whether lights with castshadow cast ray-traced shadows.
    shadow_light_fraction: fraction of direct light reaching occluded points.
    geom_metallic: resolved metallic                                   (*, ngeom)
    geom_roughness: resolved perceptual roughness                      (*, ngeom)
    geom_reflectance: dielectric reflectance, F0 = 0.16 r²             (*, ngeom)
    mat_reflectance: MuJoCo mirror reflectance per material            (*, nmat)
    light_intensity: resolved physical intensity (lux or candela)      (*, nlight)
    light_range: physical falloff radius                               (*, nlight)
    light_bulbradius: radius of the light bulb for soft shadows        (*, nlight)
    headlight_intensity: headlight illuminance times exposure (physical light model; the kernels
      divide it by each world's exposure, so the headlight stays display-relative)
    exposure: linear exposure multiplier                               (*,)
    ambient_radiance: constant environment radiance                    (*,)
    dfg_lut: split-sum DFG lookup table indexed by (NoV, perceptual roughness)
    background_linear: linear display colour of rays that hit nothing
    has_env: whether an environment map lights the scene (else constant ambient)
    env_background: whether rays that hit nothing show the environment map
    env_maps: EnvironmentMap objects behind the env arrays (keeps them alive)
    env_radiance: radiance texture per environment map                 (nenv,)
    env_prefiltered: GGX-prefiltered stack per environment map         (nenv,)
    env_sh: SH9 Lambertian radiance coefficients                       (nenv, 9)
    env_texel_angle: angular size of a radiance texel (largest map)
    env_id: environment map shown in each world                        (*,)
    env_intensity: radiance scale of the environment map               (*,)
    env_yaw: rotation of the environment map about +z, radians         (*,)
    env_prob: environment importance-sampling cell probabilities       (nenv, h, w)
    env_marginal: row CDF of env_prob                                  (nenv, h + 1)
    env_conditional: per-row column CDFs of env_prob                   (nenv, h, w + 1)
    textures: material textures indexed by MuJoCo texture id
    textures_registry: the wp.Texture2D objects behind `textures` (keeps them alive)
    tex_size: texture (width, height)                                  (ntex,)
    features: SurfaceFeatures the kernel is specialised on
    samples_per_pixel: sub-pixel samples per axis (N×N per pixel)
    shadow_samples: shadow rays per light (> 1 gives soft shadows from bulbradius)
    has_transparency: whether translucent surfaces are composited. Derived at creation from the
      rendered geoms' rgba, materials and textures: set it to True after making geoms translucent
      (e.g. replacing m.geom_rgba with alpha < 1), otherwise alpha is ignored
    max_layers: translucent layers composited per ray
    has_reflection: whether MuJoCo reflectance is rendered. Derived at creation from the rendered
      materials' reflectance (and enable_reflections): set it to True after making materials
      reflective (e.g. replacing mat_reflectance), otherwise reflectance is ignored
    render_order: RenderOrder thread-to-pixel assignment
    block_dim: threads per block of the render kernels (None: MuJoCo Warp's render block size)
    max_registers: register cap per thread for the render kernels (0: compiler default)
    ordered_traversal: nearest-first scene BVH traversal (traverse.py) instead of MuJoCo Warp's
    tier: Tier
    path_samples: path-traced samples per pixel
    max_depth: maximum path length (bounces)
    rr_depth: bounce after which Russian roulette applies
    denoise: whether the path-traced image is denoised
    denoise_iterations: à-trous filter passes
    denoise_sigma_color: luminance tolerance in standard deviations of the local noise
    denoise_sigma_normal: exponent of the normal similarity weight
    denoise_sigma_depth: relative depth tolerance per pixel of filter step
    denoise_sigma_albedo: albedo difference tolerance
    aov_albedo: first-hit albedo, the denoiser's demodulation         (nworld, npixel)
    aov_normal: first-hit shading normal (zero for the background)     (nworld, npixel)
    aov_depth: first-hit camera depth                                  (nworld, npixel)
    aov_emission: first-hit emitted radiance, exposed                  (nworld, npixel)
    aov_variance: variance of the pixel's mean exposed luminance       (nworld, npixel)
    denoise_buffers: ping-pong filter buffers                          (nworld, npixel)
    render_hdr: whether linear HDR (pre-exposure, pre-tone-mapping) is written
    hdr_data: linear HDR radiance                                      (nworld, npixel)
    traversal: ray traversal of the real-time tier: "software" (MuJoCo Warp's BVH in Warp) or
      "optix" (hardware ray tracing through OptiX, see optix/)
    optix_rebuild_every: rebuild the OptiX IAS every this many frames instead of refitting it
      (0: refit only)
    optix_max_registers: register cap of the OptiX programs (0: OptiX's default; None: 96 with
      mirror reflections, whose second trace and shading site spill at 64, else 64). Measured
      on Blackwell RTX: 64 is up to 20% faster than OptiX's default; with reflections 96 is
      1.4x faster than 64
    optix_graph_capture: allow rendering inside CUDA graph capture with traversal="optix"
      (Linux). The graph runs the OptiX calls through host nodes and a worker thread
      (optix/graph.py). After launching such a graph, synchronise (wp.synchronize(),
      torch.cuda.synchronize(), ...) before any synchronous copy to pageable host memory (e.g.
      array.numpy()): such a copy, issued while the graph waits for the worker, deadlocks.
    mesh_bvh_frame: per-mesh rotation R of the mesh BVHs (software traversal) or triangle
      GASes (OptiX), built over `R v` (None when all are in MuJoCo's mesh frame)  (nmesh,)
    trace_mesh_bvh_id: mesh BVHs of the software traversal (`base.mesh_bvh_id`, with the
      re-framed copies where mesh_bvh_frame is not the identity; None with OptiX) (nmesh,)
    trace_mesh_registry: the re-framed wp.Mesh objects (keeps them alive)
    geom_xmat_trace: geom frames of ray queries, refreshed every render  (nworld, ngeom)
    baked: bake groups compiled into the kernels as constants (`kernel_inputs.BAKE_GROUPS`; the
      other settings are read at runtime). Changing a baked value recompiles the kernel on the
      next render
    _megakernel: cached specialised render kernel
    _megakernel_key: what `_megakernel` is specialised on (`kernel_inputs.kernel_key`)
    _optix: cached OptiX acceleration structures and pipeline (traversal="optix")
    _host_cache: host-side values derived from device arrays, keyed by array identity
  """

  base: RenderContext
  light_model: int
  tone_mapper: int
  use_shadows: bool
  shadow_light_fraction: float
  geom_metallic: wp.array2d[float]
  geom_roughness: wp.array2d[float]
  geom_reflectance: wp.array2d[float]
  mat_reflectance: wp.array2d[float]
  light_intensity: wp.array2d[float]
  light_range: wp.array2d[float]
  light_bulbradius: wp.array2d[float]
  headlight_intensity: float
  exposure: wp.array[float]
  ambient_radiance: wp.array[wp.vec3]
  dfg_lut: wp.Texture2D
  background_linear: wp.vec3
  has_env: bool
  env_background: bool
  env_maps: list
  env_radiance: wp.array[wp.Texture2D]
  env_prefiltered: wp.array[wp.Texture3D]
  env_sh: wp.array2d[wp.vec3]
  env_texel_angle: float
  env_id: wp.array[int]
  env_intensity: wp.array[float]
  env_yaw: wp.array[float]
  env_prob: wp.array3d[float]
  env_marginal: wp.array2d[float]
  env_conditional: wp.array3d[float]
  textures: wp.array[wp.Texture2D]
  textures_registry: list
  tex_size: wp.array[wp.vec2]
  features: "SurfaceFeatures"
  samples_per_pixel: int
  shadow_samples: int
  has_transparency: bool
  max_layers: int
  has_reflection: bool
  render_order: int
  block_dim: Optional[int]
  max_registers: int
  ordered_traversal: bool
  tier: int
  path_samples: int
  max_depth: int
  rr_depth: int
  denoise: bool
  denoise_iterations: int
  aov_albedo: wp.array2d[wp.vec3h]
  aov_normal: wp.array2d[wp.vec3h]
  aov_depth: wp.array2d[float]
  aov_emission: wp.array2d[wp.vec3]
  aov_variance: wp.array2d[float]
  denoise_buffers: tuple[wp.array2d[wp.vec4], wp.array2d[wp.vec4]]
  render_hdr: bool
  hdr_data: wp.array2d[wp.vec3]
  traversal: str = "software"
  denoise_sigma_color: float = 4.0
  denoise_sigma_normal: float = 128.0
  denoise_sigma_depth: float = 0.02
  denoise_sigma_albedo: float = 0.05
  optix_rebuild_every: int = 0
  optix_max_registers: Optional[int] = None
  optix_graph_capture: bool = False
  mesh_bvh_frame: Optional[wp.array[wp.mat33]] = None
  trace_mesh_bvh_id: Optional[wp.array[wp.uint64]] = None
  trace_mesh_registry: Optional[list[wp.Mesh]] = None
  geom_xmat_trace: Optional[wp.array2d[wp.mat33]] = None
  baked: frozenset[str] = DEFAULT_BAKED
  _megakernel: Optional[wp.Kernel] = None
  _megakernel_key: Optional[tuple] = None
  _optix: Any = None
  _host_cache: dict[str, Any] = dataclasses.field(default_factory=dict)
