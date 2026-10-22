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

"""Per-camera views of the render outputs."""

import mujoco_warp as mjw
import warp as wp

from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def get_rgb(rc: PbrRenderContext, camera_index: int, rgb_out: wp.array3d[wp.vec3]):
  """Writes display RGB in [0, 1] for one camera into rgb_out (nworld, height, width)."""
  mjw.get_rgb(rc.base, camera_index, rgb_out)


def get_depth(rc: PbrRenderContext, camera_index: int, depth_scale: float, depth_out: wp.array3d[float]):
  """Writes planar depth / depth_scale clamped to [0, 1] for one camera."""
  mjw.get_depth(rc.base, camera_index, depth_scale, depth_out)


def get_segmentation(rc: PbrRenderContext, camera_index: int, seg_out: wp.array3d[wp.vec2i]):
  """Writes per-pixel (object id, object type) for one camera; background is (-1, -1)."""
  mjw.get_segmentation(rc.base, camera_index, seg_out)


@wp.kernel
def _extract_hdr(
  # In:
  hdr_data: wp.array2d[wp.vec3],
  rgb_adr: wp.array[int],
  camera_index: int,
  # Out:
  hdr_out: wp.array3d[wp.vec3],
):
  worldid, pixelid = wp.tid()
  xid = pixelid % hdr_out.shape[2]
  yid = pixelid // hdr_out.shape[2]
  hdr_out[worldid, yid, xid] = hdr_data[worldid, rgb_adr[camera_index] + pixelid]


def get_hdr(rc: PbrRenderContext, camera_index: int, hdr_out: wp.array3d[wp.vec3]):
  """Writes linear HDR radiance (before exposure) for one camera; needs render_hdr=True."""
  if not rc.render_hdr:
    raise ValueError("create the render context with render_hdr=True to read HDR output")
  wp.launch(
    _extract_hdr,
    dim=(hdr_out.shape[0], hdr_out.shape[1] * hdr_out.shape[2]),
    inputs=[rc.hdr_data, rc.base.rgb_adr, camera_index],
    outputs=[hdr_out],
  )
