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

"""Per-pixel camera rays (set_camera_rays) and the first-hit guides (render_guides)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr

_XML = """
  <mujoco>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".8 .8 .8" width="64" height="64"/>
      <material name="floor" texture="grid" texrepeat="4 4"/>
      <material name="m" rgba="0.8 0.3 0.2 1" metallic="0.5" roughness="0.4"/>
    </asset>
    <worldbody>
      <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
      <geom type="plane" size="2 2 0.1" material="floor"/>
      <geom type="sphere" pos="0 0 0.3" size="0.3" material="m"/>
      <geom type="box" pos="0.6 0.2 0.2" size="0.15 0.2 0.2" rgba="0.2 0.8 0.3 1"/>
      <camera name="front" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
      <camera name="side" pos="1.5 -1.5 1.2" xyaxes="0.7 0.7 0 -0.3 0.3 0.9" fovy="70"/>
    </worldbody>
  </mujoco>
"""
_RES = (24, 16)
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def pinhole_rays(width: int, height: int, fovy_deg: float, offset: float = 0.5) -> np.ndarray:
  """(height, width, 2, 3) camera-frame rays of a pinhole camera through (px, py) + offset."""
  h = np.tan(np.radians(fovy_deg) / 2.0)
  u = (np.arange(width) + offset) / width - 0.5
  v = (np.arange(height) + offset) / height - 0.5
  d = np.stack(np.broadcast_arrays(u[None, :] * 2.0 * h * width / height, -v[:, None] * 2.0 * h, -np.ones((1, 1))), axis=-1)
  d /= np.linalg.norm(d, axis=-1, keepdims=True)
  return np.stack([np.zeros_like(d), d], axis=2).astype(np.float32)


def _pinhole_bundle(fovys=(50.0, 70.0)) -> np.ndarray:
  return np.stack([pinhole_rays(*_RES, fovy) for fovy in fovys])


def _render(scene, rays=None, nworld: int = 2, **kwargs):
  s = scene(_XML, nworld=nworld, cam_res=_RES, render_depth=True, render_seg=True, camera_rays=rays, **kwargs)
  s.render()
  base = s.rc.base
  # .copy(): CPU arrays share their memory with .numpy().
  return s, base.rgb_data.numpy().copy(), base.depth_data.numpy().copy(), base.seg_data.numpy().copy()


def _channels(rgb: np.ndarray) -> np.ndarray:
  return np.stack([np.right_shift(rgb, s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.int32)


def _ray_scale(offset: float, fovys=(50.0, 70.0)) -> np.ndarray:
  """Distance along the ray per unit of planar depth, both cameras concatenated."""
  return np.concatenate([1.0 / -pinhole_rays(*_RES, f, offset)[..., 1, 2].ravel() for f in fovys])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
  "options",
  [{}, {"samples_per_pixel": 2}, {"tier": mjpbr.Tier.PATH, "path_samples": 4}],
  ids=["realtime", "supersampled", "path"],
)
def test_pinhole_rays_match_pinhole_cameras(scene, options, device):
  with wp.ScopedDevice(device):
    _, rgb, depth, seg = _render(scene, **options)
    _, rgb_rays, dist_rays, seg_rays = _render(scene, rays=_pinhole_bundle(), **options)
  assert np.abs(_channels(rgb) - _channels(rgb_rays)).max() <= 1
  np.testing.assert_array_equal(seg, seg_rays)
  # Bundle depth is the distance along the ray, pinhole depth is planar; both from the first
  # sample (the top-left one of the 2 x 2 supersampled ones).
  offset = 0.25 if options.get("samples_per_pixel") == 2 else 0.5
  np.testing.assert_allclose(dist_rays, depth * _ray_scale(offset), rtol=1e-4, atol=1e-6)


def test_rays_are_updated_and_removed(scene):
  w, h = _RES

  def first_camera(array: np.ndarray) -> np.ndarray:
    return array[0, : w * h].reshape(h, w, *array.shape[2:])

  with wp.ScopedDevice("cpu"):
    s, rgb, depth, seg = _render(scene, nworld=1)
    rays = _pinhole_bundle()
    mjpbr.set_camera_rays(s.rc, rays)
    s.render()
    assert np.abs(_channels(s.rc.base.rgb_data.numpy()) - _channels(rgb)).max() <= 1
    # Mirrored rays (x -> -x) mirror the image.
    flipped = rays.copy()
    flipped[..., 0] *= -1.0
    mjpbr.set_camera_rays(s.rc, [flipped[0], flipped[1]])
    s.render()
    np.testing.assert_array_equal(first_camera(s.rc.base.seg_data.numpy())[:, ::-1], first_camera(seg))
    scale = first_camera(_ray_scale(0.5)[None])
    np.testing.assert_allclose(first_camera(s.rc.base.depth_data.numpy())[:, ::-1], first_camera(depth) * scale, rtol=1e-4)
    mjpbr.set_camera_rays(s.rc, None)
    s.render()
    np.testing.assert_array_equal(s.rc.base.rgb_data.numpy(), rgb)


def test_rays_without_direction_miss(scene):
  rays = _pinhole_bundle()
  rays[:, :6, :, 1] = 0.0  # the top 6 rows of both cameras have no ray
  with wp.ScopedDevice("cpu"):
    _, _, _, reference = _render(scene, nworld=1)
    _, _, depth, seg = _render(scene, rays=rays, nworld=1)
  w, h = _RES
  for cam in range(2):
    pixels = slice(cam * w * h, (cam + 1) * w * h)
    image_seg = seg[0, pixels].reshape(h, w, 2)
    assert np.all(image_seg[:6] == -1) and np.all(depth[0, pixels].reshape(h, w)[:6] == 0.0)
    np.testing.assert_array_equal(image_seg[6:], reference[0, pixels].reshape(h, w, 2)[6:])


def test_parallel_rays_measure_distance_from_their_origin(scene):
  # Orthographic bundle: origins on a 1 m grid in front of the camera, all looking along -z.
  w, h = _RES
  x = ((np.arange(w) + 0.5) / w - 0.5) * 1.5
  y = (0.5 - (np.arange(h) + 0.5) / h) * 1.0
  origin = np.stack(np.broadcast_arrays(x[None, :], y[:, None], np.full((1, 1), -0.25)), axis=-1)
  direction = np.broadcast_to([0.0, 0.0, -1.0], origin.shape)
  ortho = np.stack([origin, direction], axis=2).astype(np.float32)
  xml = _XML.replace('<camera name="side" pos="1.5 -1.5 1.2" xyaxes="0.7 0.7 0 -0.3 0.3 0.9" fovy="70"/>', "")
  xml = xml.replace('pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9" fovy="50"', 'pos="0 0 3" xyaxes="1 0 0 0 1 0"')
  with wp.ScopedDevice("cpu"):
    s = scene(xml, nworld=1, cam_res=_RES, render_depth=True, render_seg=True, camera_rays=ortho[None])
    s.render()
  depth = s.rc.base.depth_data.numpy()[0].reshape(h, w)
  seg = s.rc.base.seg_data.numpy()[0].reshape(h, w, 2)
  floor = seg[..., 0] == 0
  assert floor.any()
  # Camera at z = 3, origins 0.25 m in front of it: the floor (z = 0) is 2.75 m along every ray.
  np.testing.assert_allclose(depth[floor], 2.75, atol=1e-4)
  sphere = seg[..., 0] == 1
  np.testing.assert_allclose(depth[sphere].min(), 3.0 - 0.25 - 0.6, atol=0.02)


@pytest.mark.parametrize("device", _DEVICES)
def test_guides(scene, device):
  with wp.ScopedDevice(device):
    s, *_ = _render(scene, nworld=1, render_guides=True)
    w, h = _RES
    albedo = wp.zeros((1, h, w), dtype=wp.vec3)
    normal = wp.zeros((1, h, w), dtype=wp.vec3)
    mjpbr.get_albedo(s.rc, 0, albedo)
    mjpbr.get_normal(s.rc, 0, normal)
  seg = s.rc.base.seg_data.numpy()[0, : w * h].reshape(h, w, 2)
  albedo_np = albedo.numpy()[0]
  normal_np = normal.numpy()[0]
  box = seg[..., 0] == 2
  assert box.any()
  np.testing.assert_allclose(albedo_np[box], np.broadcast_to([0.2, 0.8, 0.3], albedo_np[box].shape), atol=1e-5)
  floor = seg[..., 0] == 0
  np.testing.assert_allclose(normal_np[floor], np.broadcast_to([0.0, 0.0, 1.0], normal_np[floor].shape), atol=1e-5)
  sphere = seg[..., 0] == 1
  np.testing.assert_allclose(np.linalg.norm(normal_np[sphere], axis=-1), 1.0, atol=1e-4)


def test_guides_need_render_guides(scene):
  with wp.ScopedDevice("cpu"):
    s = scene(_XML, cam_res=_RES)
    with pytest.raises(ValueError, match="render_guides"):
      mjpbr.get_albedo(s.rc, 0, wp.zeros((1, _RES[1], _RES[0]), dtype=wp.vec3))


def test_ray_validation(scene):
  with wp.ScopedDevice("cpu"):
    s = scene(_XML, cam_res=_RES)
    with pytest.raises(ValueError, match="each of the 2 rendering cameras"):
      mjpbr.set_camera_rays(s.rc, _pinhole_bundle()[:1])
    with pytest.raises(ValueError, match=r"shape \(16, 24, 2\)"):
      mjpbr.set_camera_rays(s.rc, [pinhole_rays(24, 16, 50.0), pinhole_rays(16, 16, 50.0)])
    with pytest.raises(ValueError, match="dtype vec3"):
      wrong: Any = wp.zeros((2, 16, 24, 2), dtype=wp.vec4)
      mjpbr.set_camera_rays(s.rc, wrong)
    assert s.rc.ray_direction is None
