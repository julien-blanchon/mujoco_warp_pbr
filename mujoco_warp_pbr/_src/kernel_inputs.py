# Copyright 2026 The mujoco_warp_pbr Authors
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

"""Kernel inputs shared by the real-time and path-traced tiers, grouped into structs."""

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model

from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class Cameras:
  # Model:
  cam_projection: wp.array[int]
  cam_fovy: wp.array2d[float]
  cam_sensorsize: wp.array[wp.vec2]
  cam_intrinsic: wp.array2d[wp.vec4]
  # Data:
  cam_xpos: wp.array2d[wp.vec3]
  cam_xmat: wp.array2d[wp.mat33]
  # Render context:
  nrender: int
  cam_res: wp.array[wp.vec2i]
  cam_id_map: wp.array[int]
  # Runtime loop bounds keep the sample loop rolled (constant bounds would be unrolled).
  samples_per_axis: int


@wp.struct
class Environment:
  exposure: wp.array[float]
  ambient_radiance: wp.array[wp.vec3]
  background: wp.vec3
  dfg_lut: wp.Texture2D
  env_radiance: wp.array[wp.Texture2D]
  env_prefiltered: wp.array[wp.Texture3D]
  env_sh: wp.array2d[wp.vec3]
  env_id: wp.array[int]
  env_intensity: wp.array[float]
  env_yaw: wp.array[float]
  env_prob: wp.array3d[float]
  env_marginal: wp.array2d[float]
  env_conditional: wp.array3d[float]
  max_layers: int


@wp.struct
class Outputs:
  render_rgb: wp.array[bool]
  render_depth: wp.array[bool]
  render_seg: wp.array[bool]
  rgb_adr: wp.array[int]
  depth_adr: wp.array[int]
  seg_adr: wp.array[int]
  rgb: wp.array2d[wp.uint32]
  depth: wp.array2d[float]
  seg: wp.array2d[wp.vec2i]
  hdr: wp.array2d[wp.vec3h]


@wp.func
def rand2(seed: int, index: int) -> wp.vec2:
  return wp.vec2(wp.randf(wp.rand_init(seed, 2 * index)), wp.randf(wp.rand_init(seed, 2 * index + 1)))


def cameras(m: Model, d: Data, rc: PbrRenderContext) -> Cameras:
  cams = Cameras()
  cams.cam_projection = m.cam_projection
  cams.cam_fovy = m.cam_fovy
  cams.cam_sensorsize = m.cam_sensorsize
  cams.cam_intrinsic = m.cam_intrinsic
  cams.cam_xpos = d.cam_xpos
  cams.cam_xmat = d.cam_xmat
  cams.nrender = rc.base.nrender
  cams.cam_res = rc.base.cam_res
  cams.cam_id_map = rc.base.cam_id_map
  cams.samples_per_axis = rc.samples_per_pixel
  return cams


def environment(rc: PbrRenderContext) -> Environment:
  env = Environment()
  env.exposure = rc.exposure
  env.ambient_radiance = rc.ambient_radiance
  env.background = rc.background_linear
  env.dfg_lut = rc.dfg_lut
  env.env_radiance = rc.env_radiance
  env.env_prefiltered = rc.env_prefiltered
  env.env_sh = rc.env_sh
  env.env_id = rc.env_id
  env.env_intensity = rc.env_intensity
  env.env_yaw = rc.env_yaw
  env.env_prob = rc.env_prob
  env.env_marginal = rc.env_marginal
  env.env_conditional = rc.env_conditional
  env.max_layers = rc.max_layers if rc.has_transparency else 1
  return env


def outputs(rc: PbrRenderContext) -> Outputs:
  base = rc.base
  out = Outputs()
  out.render_rgb = base.render_rgb
  out.render_depth = base.render_depth
  out.render_seg = base.render_seg
  out.rgb_adr = base.rgb_adr
  out.depth_adr = base.depth_adr
  out.seg_adr = base.seg_adr
  out.rgb = base.rgb_data
  out.depth = base.depth_data
  out.seg = base.seg_data
  out.hdr = rc.hdr_data
  return out
