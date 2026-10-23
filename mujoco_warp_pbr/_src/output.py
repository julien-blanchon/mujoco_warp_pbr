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

"""Per-camera views of the render outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjw
import numpy as np
import warp as wp

from mujoco_warp_pbr._src.types import PbrRenderContext

if TYPE_CHECKING:
  from shape_extensions import IntVar

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def get_rgb[NWorld: IntVar, NOut: IntVar, H: IntVar, W: IntVar](
  rc: PbrRenderContext[NWorld], camera_index: int, rgb_out: wp.array[wp.vec3, [NOut, H, W]]
) -> None:
  """Writes display RGB in [0, 1] for one camera into rgb_out (nworld, height, width)."""
  mjw.get_rgb(rc.base, camera_index, rgb_out)


def get_depth[NWorld: IntVar, NOut: IntVar, H: IntVar, W: IntVar](
  rc: PbrRenderContext[NWorld], camera_index: int, depth_scale: float, depth_out: wp.array[float, [NOut, H, W]]
) -> None:
  """Writes planar depth / depth_scale clamped to [0, 1] for one camera."""
  mjw.get_depth(rc.base, camera_index, depth_scale, depth_out)


def get_segmentation[NWorld: IntVar, NOut: IntVar, H: IntVar, W: IntVar](
  rc: PbrRenderContext[NWorld], camera_index: int, seg_out: wp.array[wp.vec2i, [NOut, H, W]]
) -> None:
  """Writes per-pixel (object id, object type) for one camera; background is (-1, -1)."""
  mjw.get_segmentation(rc.base, camera_index, seg_out)


@wp.kernel
def _extract_hdr(
  # In:
  hdr_data: wp.array2d[wp.vec3],
  rgb_adr: wp.array1d[int],
  camera_index: int,
  # Out:
  hdr_out: wp.array3d[wp.vec3],
):
  worldid, pixelid = wp.tid()
  xid = pixelid % hdr_out.shape[2]
  yid = pixelid // hdr_out.shape[2]
  hdr_out[worldid, yid, xid] = hdr_data[worldid, rgb_adr[camera_index] + pixelid]


def _camera_res(rc: PbrRenderContext) -> np.ndarray[[int, 2], np.dtype[np.int32]]:
  """(nrender, 2) camera resolutions on host (cached per `base.cam_res` array)."""
  cam_res = rc.base.cam_res
  cached = rc._host_cache.get("cam_res")
  if cached is None or cached[0] is not cam_res:
    cached = (cam_res, cam_res.numpy())
    rc._host_cache["cam_res"] = cached
  return cached[1]


def get_hdr[NWorld: IntVar, NOut: IntVar, H: IntVar, W: IntVar](
  rc: PbrRenderContext[NWorld], camera_index: int, hdr_out: wp.array[wp.vec3, [NOut, H, W]]
) -> None:
  """Writes linear HDR radiance (before exposure) for one camera into hdr_out (<= nworld, h, w).

  Needs render_hdr=True.

  Raises:
    ValueError: without render_hdr, for a camera index out of range or an output whose shape
      does not match the camera image.
  """
  if not rc.render_hdr:
    raise ValueError("create the render context with render_hdr=True to read HDR output")
  res = _camera_res(rc)
  if not 0 <= camera_index < len(res):
    raise ValueError(f"camera_index must be in [0, {len(res)}), got {camera_index}")
  nworld = rc.hdr_data.shape[0]
  width, height = (int(v) for v in res[camera_index])
  if hdr_out.ndim != 3 or hdr_out.shape[0] > nworld or tuple(hdr_out.shape[1:]) != (height, width):
    raise ValueError(f"hdr_out must have shape (<= {nworld}, {height}, {width}), got {tuple(hdr_out.shape)}")
  wp.launch(
    _extract_hdr,
    dim=(hdr_out.shape[0], height * width),
    inputs=[rc.hdr_data, rc.base.rgb_adr, camera_index],
    outputs=[hdr_out],
  )
