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

"""Thread -> camera pixel assignment and camera rays, shared by both tiers.

Camera rays come from one of two sources, chosen when the kernels are built (`make_camera_rays`):
  * pinhole (default): MuJoCo Warp's `compute_ray` from the camera's fovy / intrinsics;
  * ray bundle (`PbrRenderContext.ray_direction` set, see `rays.set_camera_rays`): per-pixel
    camera-frame origins and directions, sampled at pixel centres. A sub-pixel sample
    interpolates bilinearly between the four nearest bundle rays (extrapolating linearly past
    the outer pixel centres): directions on the image plane z = -1 when the four point forward
    (exact for pinhole bundles), else on the unit sphere, then renormalised. A bundle ray with a
    zero direction marks a pixel without a ray, rendered as a miss; samples next to it take the
    nearest ray.
    Ray cones (texture filtering, specular anti-aliasing) take their spread and width from the
    differences between neighbouring bundle rays.
"""

from collections.abc import Callable

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
# Depth axis of bundle rays: depth = dist * -(0, 0, -1).z, the distance along the ray.
_ALONG_RAY = wp.constant(wp.vec3(0.0, 0.0, -1.0))


@wp.func
def thread_ids(order: int, i: int, j: int) -> tuple[int, int]:
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
def camera_frame_ray(pix: Pixel, subdiv: int, sx: int, sy: int, znear: float) -> tuple[wp.vec3, wp.vec3]:
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
def camera_ray(pix: Pixel, subdiv: int, sx: int, sy: int, znear: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """(world origin, world direction, camera-frame direction) of `camera_frame_ray`."""
  ray_dir_cam, ray_offset_cam = camera_frame_ray(pix, subdiv, sx, sy, znear)
  return pix.xpos + pix.xmat @ ray_offset_cam, pix.xmat @ ray_dir_cam, ray_dir_cam


@wp.func
def max_ray_dist(ray_dir_cam: wp.vec3, zfar: float) -> float:
  """Distance to the far plane along a camera ray (unbounded when zfar <= 0)."""
  if zfar > 0.0:
    return zfar / wp.max(-ray_dir_cam[2], wp.static(_MIN_FORWARD))
  return float(MJ_MAXVAL)


@wp.func
def _bundle_axis(i: int, n: int, nb: int) -> tuple[int, int, float]:
  """Bundle rays (x0, x1) and weight t of sample i of n along an image axis of nb bundle rays.

  Sample i sits at image coordinate (i + 0.5) / n, i.e. bundle coordinate x = x0 + t with
  x = (i + 0.5) nb / n - 0.5 = num / den. Integer arithmetic keeps t exactly 0 when the sample
  falls on a bundle pixel centre (e.g. one sample per pixel at the bundle's resolution).
  """
  num = wp.int64(2 * i + 1) * wp.int64(nb) - wp.int64(n)
  den = wp.int64(2 * n)
  # num > -den: floor division of the positive num + den.
  x0 = wp.clamp(int((num + den) // den) - 1, 0, wp.max(nb - 2, 0))  # pyrefly: ignore[unsupported-operation]  # stubs: no int64 //
  x1 = wp.min(x0 + 1, nb - 1)
  return x0, x1, float(num - wp.int64(x0) * den) / float(den)


@wp.func
def _lerp(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
  return a + (b - a) * t


@wp.func
def bundle_ray(cams: Cameras, pix: Pixel, subdiv: int, sx: int, sy: int) -> tuple[wp.vec3, wp.vec3]:
  """(direction, origin) in the camera frame of the bundle ray through sub-pixel cell (sx, sy).

  The direction is normalised, or zero where the pixel has no ray.
  """
  res = cams.ray_res[pix.camid]
  adr = cams.ray_adr[pix.camid]
  x0, x1, tx = _bundle_axis(pix.px * subdiv + sx, pix.width * subdiv, res[0])
  y0, y1, ty = _bundle_axis(pix.py * subdiv + sy, pix.height * subdiv, res[1])
  r0 = adr + y0 * res[0]
  r1 = adr + y1 * res[0]
  d00 = cams.ray_direction[r0 + x0]
  d10 = cams.ray_direction[r0 + x1]
  d01 = cams.ray_direction[r1 + x0]
  d11 = cams.ray_direction[r1 + x1]
  if wp.length_sq(d00) == 0.0 or wp.length_sq(d10) == 0.0 or wp.length_sq(d01) == 0.0 or wp.length_sq(d11) == 0.0:
    # Next to a pixel without a ray: the nearest bundle ray, no interpolation.
    near = wp.where(ty < 0.5, r0, r1) + wp.where(tx < 0.5, x0, x1)
    direction = cams.ray_direction[near]
    if wp.length_sq(direction) > 0.0:
      direction = wp.normalize(direction)
    return direction, cams.ray_origin[near]
  o00 = cams.ray_origin[r0 + x0]
  o10 = cams.ray_origin[r0 + x1]
  o01 = cams.ray_origin[r1 + x0]
  o11 = cams.ray_origin[r1 + x1]
  if d00[2] < 0.0 and d10[2] < 0.0 and d01[2] < 0.0 and d11[2] < 0.0:
    # Forward rays: interpolate on the image plane z = -1 (exact for pinhole bundles).
    d00 = d00 * (-1.0 / d00[2])
    d10 = d10 * (-1.0 / d10[2])
    d01 = d01 * (-1.0 / d01[2])
    d11 = d11 * (-1.0 / d11[2])
  direction = _lerp(_lerp(d00, d10, tx), _lerp(d01, d11, tx), ty)
  origin = _lerp(_lerp(o00, o10, tx), _lerp(o01, o11, tx), ty)
  if wp.length_sq(direction) > 0.0:
    direction = wp.normalize(direction)
  return direction, origin


@wp.func
def _bundle_step(cams: Cameras, a: int, b: int, extrapolate: bool) -> tuple[float, float]:
  """(angle, distance) between the directions and the origins of bundle rays a and b.

  With `extrapolate`, between ray b and its linear extrapolation beyond b instead: on the image
  plane z = -1 when both rays point forward (exact for pinhole bundles), else on the sphere.
  """
  da = cams.ray_direction[a]
  db = cams.ray_direction[b]
  if wp.length_sq(da) == 0.0 or wp.length_sq(db) == 0.0:
    return 0.0, 0.0
  da = wp.normalize(da)
  db = wp.normalize(db)
  width = wp.length(cams.ray_origin[b] - cams.ray_origin[a])
  if extrapolate:
    if da[2] < 0.0 and db[2] < 0.0:
      return wp.length(wp.normalize(db * (-2.0 / db[2]) + da * (1.0 / da[2])) - db), width
    return wp.length(wp.normalize(db * 2.0 - da) - db), width
  return wp.length(db - da), width


@wp.func
def bundle_cone(cams: Cameras, pix: Pixel) -> tuple[float, float]:
  """(spread, width) of one image pixel: ray difference to the next bundle ray along x.

  As for pinhole cameras, the neighbour is the next pixel of the row (in the last column, the
  linear extrapolation of the previous one). Differences are scaled from bundle pixels to image
  pixels (the image may be rendered at a multiple of the bundle's resolution).
  """
  res = cams.ray_res[pix.camid]
  if res[0] < 2:
    return 0.0, 0.0
  bx = (pix.px * res[0]) // pix.width
  b = cams.ray_adr[pix.camid] + ((pix.py * res[1]) // pix.height) * res[0] + bx
  spread = float(0.0)
  width = float(0.0)
  if bx + 1 < res[0]:
    spread, width = _bundle_step(cams, b, b + 1, False)
  else:
    spread, width = _bundle_step(cams, b - 1, b, True)
  scale = float(res[0]) / float(pix.width)
  return spread * scale, width * scale


def make_camera_rays(bundle: bool) -> tuple[Callable, Callable]:
  """(primary_ray, pixel_cone) funcs of the pinhole cameras, or of a ray bundle.

  primary_ray(cams, pix, subdiv, sx, sy, s) -> (world origin, world direction, max distance,
  depth axis): the ray through sub-pixel cell (sx, sy) of the pixel divided into subdiv × subdiv
  cells (subdiv 1 and (0, 0): the pixel centre). Depth is hit distance * -depth_axis.z: MuJoCo
  Warp's planar depth for pinhole cameras, the distance along the ray for bundles (whose rays
  need not share an image plane). Likewise the far plane `zfar` is planar for pinhole cameras
  and a distance along the ray for bundles; a bundle pixel without a ray gets max distance 0
  (nothing is hit).

  pixel_cone(cams, pix, s) -> (spread, width): angular spread and width (orthographic origins)
  of one pixel, the ray-cone parameters.
  """
  if not bundle:

    @wp.func
    def primary_ray(
      cams: Cameras, pix: Pixel, subdiv: int, sx: int, sy: int, s: Settings
    ) -> tuple[wp.vec3, wp.vec3, float, wp.vec3]:
      origin, direction, ray_dir_cam = camera_ray(pix, subdiv, sx, sy, s.znear)
      return origin, direction, max_ray_dist(ray_dir_cam, s.zfar), ray_dir_cam

    @wp.func
    def pixel_cone(cams: Cameras, pix: Pixel, s: Settings) -> tuple[float, float]:
      d0, o0 = camera_frame_ray(pix, 1, 0, 0, s.znear)
      d1, o1 = camera_frame_ray(pix, 1, 1, 0, s.znear)
      return wp.length(d1 - d0), wp.length(o1 - o0)

    return primary_ray, pixel_cone

  @wp.func
  def bundle_primary_ray(
    cams: Cameras, pix: Pixel, subdiv: int, sx: int, sy: int, s: Settings
  ) -> tuple[wp.vec3, wp.vec3, float, wp.vec3]:
    ray_dir_cam, ray_org_cam = bundle_ray(cams, pix, subdiv, sx, sy)
    max_dist = float(MJ_MAXVAL)
    if s.zfar > 0.0:
      max_dist = s.zfar
    if wp.length_sq(ray_dir_cam) == 0.0:
      max_dist = 0.0
      ray_dir_cam = _ALONG_RAY
    return pix.xpos + pix.xmat @ ray_org_cam, pix.xmat @ ray_dir_cam, max_dist, _ALONG_RAY

  @wp.func
  def bundle_pixel_cone(cams: Cameras, pix: Pixel, s: Settings) -> tuple[float, float]:
    return bundle_cone(cams, pix)

  return bundle_primary_ray, bundle_pixel_cone


def make_geometry_only(trace_closest: Callable, primary_ray: Callable) -> Callable:
  """geometry_only(geo, cams, out, pix, s, worldid, subdiv, centre): depth / segmentation.

  Traces the primary ray (`make_camera_rays`) through sub-pixel cell (centre, centre) without
  shading, for cameras that render no RGB.
  """

  @wp.func
  def geometry_only(
    geo: Geometry, cams: Cameras, out: Outputs, pix: Pixel, s: Settings, worldid: int, subdiv: int, centre: int
  ):
    origin, direction, max_dist, depth_axis = primary_ray(cams, pix, subdiv, centre, centre, s)
    hit = trace_closest(geo, worldid, origin, direction, max_dist)
    write_geometry(out, worldid, pix.camid, pix.pixelid, hit, depth_axis, pix.want_depth, pix.want_seg)

  return geometry_only
