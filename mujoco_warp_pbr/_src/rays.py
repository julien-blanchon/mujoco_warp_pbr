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

"""Per-pixel camera rays (ray bundles) replacing the cameras' pinhole projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from mujoco_warp_pbr._src.output import camera_res

if TYPE_CHECKING:
  from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# One camera's rays, (height, width, 2): camera-frame origin [..., 0] and direction [..., 1].
type CameraRayImage = wp.array[wp.vec3, [int, int, 2]]
# Rays of every rendering camera: (nrender, height, width, 2) when the cameras share a
# resolution, else one image per camera; NumPy float arrays of shape (..., 2, 3) are accepted.
type CameraRays = wp.array[wp.vec3, [int, int, int, 2]] | Sequence[CameraRayImage | np.ndarray] | np.ndarray


@wp.kernel
def _copy_rays(
  # In:
  rays: wp.array3d[wp.vec3],
  adr: int,
  # Out:
  origin: wp.array1d[wp.vec3],
  direction: wp.array1d[wp.vec3],
):
  y, x = wp.tid()
  i = adr + y * rays.shape[1] + x
  origin[i] = rays[y, x, 0]
  direction[i] = rays[y, x, 1]


def _as_images(camera_rays: CameraRays, device: Any) -> list[wp.array3d[wp.vec3]]:
  """The rays as one (height, width, 2) vec3 Warp array per camera."""
  if isinstance(camera_rays, np.ndarray):
    if camera_rays.ndim != 5 or camera_rays.shape[-2:] != (2, 3):
      raise ValueError(f"camera_rays as a NumPy array must have shape (ncam, height, width, 2, 3), got {camera_rays.shape}")
    return [_as_image(image, device) for image in camera_rays]
  if isinstance(camera_rays, wp.array):
    if camera_rays.ndim != 4 or camera_rays.shape[3] != 2:
      raise ValueError(f"camera_rays must have shape (ncam, height, width, 2), got {tuple(camera_rays.shape)}")
    return [_as_image(camera_rays[i], device) for i in range(camera_rays.shape[0])]
  return [_as_image(image, device) for image in camera_rays]


def _as_image(image: Any, device: Any) -> wp.array3d[wp.vec3]:
  if isinstance(image, np.ndarray):
    if image.ndim != 4 or image.shape[-2:] != (2, 3):
      raise ValueError(f"a camera's rays as a NumPy array must have shape (height, width, 2, 3), got {image.shape}")
    return wp.array(np.ascontiguousarray(image, dtype=np.float32), dtype=wp.vec3, device=device)
  if not isinstance(image, wp.array):
    raise TypeError(f"camera rays must be Warp or NumPy arrays, got {type(image).__name__}")
  if image.dtype != wp.vec3:
    raise ValueError(f"camera rays must have dtype vec3, got {wp.types.type_repr(image.dtype)}")
  if image.ndim != 3 or image.shape[2] != 2:
    raise ValueError(f"a camera's rays must have shape (height, width, 2), got {tuple(image.shape)}")
  if image.device != device:
    raise ValueError(f"camera rays must be on the render context's device {device}, got {image.device}")
  return image


def set_camera_rays(rc: PbrRenderContext, camera_rays: CameraRays | None) -> None:
  """Renders the context's cameras from per-pixel rays instead of their pinhole projection.

  Each rendering camera (in the order of `cam_active`) gets one ray per pixel, in its camera
  frame (MuJoCo's camera frame: -z forward, +y up, +x right, which is also Newton's), placed in
  the world by the camera's pose (`d.cam_xpos` / `d.cam_xmat`, so per world). The layout follows
  Newton's single-sample `camera_rays` bundles: ``rays[..., 0]`` is the origin and
  ``rays[..., 1]`` the direction (normalised by the renderer; a zero direction marks a pixel
  without a ray, rendered as background with depth 0).

  With rays, depth is the distance along each ray from its origin and `zfar` bounds that
  distance (both planar for pinhole cameras); znear and the cameras' fovy / intrinsics are not
  used. Sub-pixel samples (samples_per_pixel, the path tracer's jitter) interpolate between
  neighbouring rays, and texture filtering estimates each pixel's footprint from them.

  The rays are copied: call again after changing them (cheap, no recompilation). Switching
  between rays and pinhole cameras (None) recompiles the render kernels.

  Args:
    rc: The render context.
    camera_rays: An array (nrender, height, width, 2) of vec3 when every rendering camera has
      the same resolution, or a sequence of one (height, width, 2) array per rendering camera at
      its resolution (NumPy float arrays of shape (..., 2, 3) are accepted); None renders the
      pinhole cameras again.

  Raises:
    ValueError: for a camera count, resolution, dtype or device that does not match the context.
    TypeError: for rays that are not Warp or NumPy arrays.
  """
  if camera_rays is None:
    rc.ray_origin = rc.ray_direction = rc.ray_adr = rc.ray_res = None
    return
  base = rc.base
  device = base.cam_res.device
  images = _as_images(camera_rays, device)
  cam_res = camera_res(rc)
  if len(images) != base.nrender:
    raise ValueError(f"camera_rays must have rays for each of the {base.nrender} rendering cameras, got {len(images)}")
  for i, (image, (width, height)) in enumerate(zip(images, cam_res, strict=True)):
    if tuple(image.shape[:2]) != (height, width):
      raise ValueError(
        f"the rays of rendering camera {i} must have shape ({height}, {width}, 2) (its resolution), got {tuple(image.shape)}"
      )
  total = int(base.total_rays)
  npix = cam_res[:, 0].astype(np.int64) * cam_res[:, 1]
  adr = np.concatenate([[0], np.cumsum(npix)[:-1]]).astype(np.int32)
  if rc.ray_direction is None or rc.ray_direction.shape[0] != total:
    with wp.ScopedDevice(device):
      rc.ray_origin = wp.zeros(total, dtype=wp.vec3)
      rc.ray_direction = wp.zeros(total, dtype=wp.vec3)
      rc.ray_adr = wp.array(adr, dtype=int)
      rc.ray_res = wp.array(cam_res, dtype=wp.vec2i)
  for image, start in zip(images, adr, strict=True):
    wp.launch(
      _copy_rays,
      dim=(image.shape[0], image.shape[1]),
      inputs=[image, int(start)],
      outputs=[rc.ray_origin, rc.ray_direction],
      device=device,
    )
