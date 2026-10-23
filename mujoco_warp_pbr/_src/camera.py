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

"""Thread -> camera pixel assignment and camera rays, shared by both tiers."""

from typing import Callable, Tuple

import warp as wp
from mujoco_warp._src.render_util import compute_ray
from mujoco_warp._src.types import MJ_MAXVAL

from mujoco_warp_pbr._src.kernel_inputs import Cameras
from mujoco_warp_pbr._src.kernel_inputs import Outputs
from mujoco_warp_pbr._src.kernel_inputs import Settings
from mujoco_warp_pbr._src.kernel_inputs import write_geometry
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import RenderOrder

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Height of the pixel bands of RenderOrder.TILED.
TILE_ROWS = 8
# Floor of -z of camera-frame ray directions when converting the planar far plane to a distance.
_MIN_FORWARD = 1.0e-6


@wp.func
def thread_ids(order: int, i: int, j: int) -> Tuple[int, int]:
  """(worldid, rayid) of thread (i, j) = wp.tid() of a `kernel_inputs.launch_dim` launch."""
  if order == wp.static(int(RenderOrder.WORLD)):
    return j, i
  return i, j


@wp.func
def thread_pixel(order: int, index: int, width: int, height: int) -> wp.vec2i:
  """Thread index within a camera image -> (px, py), the bijection of a RenderOrder."""
  if order == wp.static(int(RenderOrder.TILED)):
    band = index // (width * wp.static(TILE_ROWS))
    rows = wp.min(wp.static(TILE_ROWS), height - band * wp.static(TILE_ROWS))
    i = index - band * width * wp.static(TILE_ROWS)
    return wp.vec2i(i // rows, band * wp.static(TILE_ROWS) + i % rows)
  return wp.vec2i(index % width, index // width)


@wp.struct
class Pixel:
  """A camera pixel rendered by one thread.

  Attributes:
    camid: rendering camera (index into the context's active cameras), -1 if nothing to do.
    pixelid: row-major pixel index within the camera image.
    px: pixel column.
    py: pixel row.
    width: image width.
    height: image height.
    want_rgb: whether the camera renders RGB.
    want_depth: whether the camera renders depth.
    want_seg: whether the camera renders segmentation.
    projection: camera projection type.
    fovy: vertical field of view.
    sensorsize: sensor size (intrinsic cameras).
    intrinsic: camera intrinsics.
    xmat: camera frame in world.
    xpos: camera position in world.
  """

  camid: int
  pixelid: int
  px: int
  py: int
  width: int
  height: int
  want_rgb: bool
  want_depth: bool
  want_seg: bool
  projection: int
  fovy: float
  sensorsize: wp.vec2
  intrinsic: wp.vec4
  xmat: wp.mat33
  xpos: wp.vec3


@wp.func
def setup_pixel(cams: Cameras, out: Outputs, s: Settings, worldid: int, rayid: int) -> Pixel:
  """The pixel of global ray `rayid` (active cameras' images concatenated) and its camera."""
  pix = Pixel()
  pix.camid = -1
  offset = int(0)
  for i in range(cams.nrender):
    npix = cams.cam_res[i][0] * cams.cam_res[i][1]
    if rayid < offset + npix:
      pix.camid = i
      pix.pixelid = rayid - offset
      break
    offset += npix
  if pix.camid == -1:
    return pix
  camid = pix.camid
  pix.want_rgb = False
  if s.any_rgb:
    pix.want_rgb = out.render_rgb[camid]
  pix.want_depth = out.render_depth[camid]
  pix.want_seg = out.render_seg[camid]
  if not pix.want_rgb and not pix.want_depth and not pix.want_seg:
    pix.camid = -1
    return pix

  cam = cams.cam_id_map[camid]
  pix.width = cams.cam_res[camid][0]
  pix.height = cams.cam_res[camid][1]
  p = thread_pixel(s.render_order, pix.pixelid, pix.width, pix.height)
  pix.px = p[0]
  pix.py = p[1]
  pix.pixelid = p[1] * pix.width + p[0]
  pix.projection = cams.cam_projection[cam]
  pix.fovy = cams.cam_fovy[worldid % cams.cam_fovy.shape[0], cam]
  pix.sensorsize = cams.cam_sensorsize[cam]
  pix.intrinsic = cams.cam_intrinsic[worldid % cams.cam_intrinsic.shape[0], cam]
  pix.xmat = cams.cam_xmat[worldid, cam]
  pix.xpos = cams.cam_xpos[worldid, cam]
  return pix


@wp.func
def headlight_dir(pix: Pixel) -> wp.vec3:
  """Direction of the headlight's light: the camera's viewing direction (-z)."""
  return -wp.vec3(pix.xmat[0, 2], pix.xmat[1, 2], pix.xmat[2, 2])


@wp.func
def camera_frame_ray(pix: Pixel, subdiv: int, sx: int, sy: int, znear: float) -> Tuple[wp.vec3, wp.vec3]:
  """(direction, origin offset) in the camera frame of the ray through sub-pixel cell (sx, sy).

  The pixel is divided into subdiv × subdiv cells; the ray goes through the centre of cell
  (sx, sy), so subdiv 1 and (0, 0) give the pixel centre (and (1, 0) the next pixel's).
  """
  return compute_ray(
    pix.projection,
    pix.fovy,
    pix.sensorsize,
    pix.intrinsic,
    pix.width * subdiv,
    pix.height * subdiv,
    pix.px * subdiv + sx,
    pix.py * subdiv + sy,
    znear,
  )


@wp.func
def camera_ray(pix: Pixel, subdiv: int, sx: int, sy: int, znear: float) -> Tuple[wp.vec3, wp.vec3, wp.vec3]:
  """(world origin, world direction, camera-frame direction) of `camera_frame_ray`."""
  ray_dir_cam, ray_offset_cam = camera_frame_ray(pix, subdiv, sx, sy, znear)
  return pix.xpos + pix.xmat @ ray_offset_cam, pix.xmat @ ray_dir_cam, ray_dir_cam


@wp.func
def max_ray_dist(ray_dir_cam: wp.vec3, zfar: float) -> float:
  """Distance to the far plane along a camera ray (unbounded when zfar <= 0)."""
  if zfar > 0.0:
    return zfar / wp.max(-ray_dir_cam[2], wp.static(_MIN_FORWARD))
  return float(MJ_MAXVAL)


def make_geometry_only(trace_closest: Callable) -> Callable:
  """geometry_only(geo, out, pix, s, worldid, subdiv, centre): depth / segmentation of a pixel.

  Traces the primary ray through sub-pixel cell (centre, centre) without shading, for cameras
  that render no RGB.
  """

  @wp.func
  def geometry_only(geo: Geometry, out: Outputs, pix: Pixel, s: Settings, worldid: int, subdiv: int, centre: int):
    origin, direction, ray_dir_cam = camera_ray(pix, subdiv, centre, centre, s.znear)
    hit = trace_closest(geo, worldid, origin, direction, max_ray_dist(ray_dir_cam, s.zfar))
    write_geometry(out, worldid, pix.camid, pix.pixelid, hit, ray_dir_cam, pix.want_depth, pix.want_seg)

  return geometry_only
