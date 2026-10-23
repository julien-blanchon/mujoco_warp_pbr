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

"""newton_pbr.SensorCamera as a drop-in for newton.sensors.SensorCamera (Warp CPU device)."""

import math
from typing import Any

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")
from newton.sensors import SensorCamera as NewtonSensorCamera  # noqa: E402

import newton_pbr  # noqa: E402
from newton_pbr import SensorCamera  # noqa: E402

# Newton's own SensorCamera tests whose scenario does not depend on Newton's shading, run against
# newton_pbr.SensorCamera.
_NEWTON_TESTS = [
  "test_update_validates_rays_and_transforms",
  "test_update_resolves_ssaa_rays",
  "test_update_resolves_msaa_rays",
  "test_update_rejects_invalid_anti_aliasing",
  "test_update_rejects_multisample_rays_without_anti_aliasing",
  "test_update_renders_from_camera_transforms",
  "test_update_respects_disable_clear_flag",
  "test_update_clears_reserved_and_out_of_range_world_indices",
  "test_update_respects_disable_preserve_flag",
  "test_update_defaults_world_indices_to_identity",
  "test_update_without_world_indices_maps_view_to_world",
  "test_world_indices_decouple_views_from_worlds",
  "test_update_rejects_default_world_indices_exceeding_world_count",
  "test_render_forward_depth_output",
  "test_forward_depth_uses_camera_origin",
  "test_update_supports_all_render_orders_with_3d_outputs",
]
_RES = 40
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
_FOV = math.radians(50.0)


@pytest.mark.parametrize("name", _NEWTON_TESTS)
def test_newton_scenario(monkeypatch, name):
  newton_tests = pytest.importorskip("newton.tests.test_sensor_camera")
  monkeypatch.setattr(newton_tests, "SensorCamera", SensorCamera)
  monkeypatch.setattr(newton_tests, "get_test_devices", lambda: ["cpu"])
  case = newton_tests.TestSensorCamera(name)
  with wp.ScopedDevice("cpu"):
    getattr(case, name)()


def _scene(nworld: int = 2, device: str = "cpu") -> tuple[Any, Any, Any]:
  """Two worlds with every primitive type and a mesh; world 1 shifted; one camera per world."""
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
  world.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, 0.45), wp.quat_identity()), radius=0.2)
  arm = world.add_body(xform=wp.transform((0.7, 0.0, 0.4), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.6)))
  world.add_shape_capsule(arm, radius=0.1, half_height=0.25, color=(0.2, 0.3, 0.9))
  verts = np.array([[0.4, 0.4, 0.0], [0.8, 0.4, 0.0], [0.4, 0.8, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
  faces = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32)
  world.add_shape_mesh(-1, mesh=newton.Mesh(verts, faces), xform=wp.transform((-1.0, -0.6, 0.0), wp.quat_identity()))
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, nworld)
  with wp.ScopedDevice(device):
    model = scene.finalize(device=device)
    state = model.state()
  q = state.body_q.numpy()
  q[len(q) // nworld :, 1] += 0.3
  state.body_q.assign(q)
  return model, state, _look_at([0.0, -3.0, 1.2], nworld, device)


def _look_at(eye: list[float], count: int, device: Any = "cpu", target=(0.0, 0.0, 0.3)) -> Any:
  eye_np = np.asarray(eye, dtype=np.float64)
  forward = np.asarray(target) - eye_np
  # numpy stubs: norm returns a bare `floating`, which no array operator accepts.
  forward /= np.linalg.norm(forward)  # pyrefly: ignore[unsupported-operation]
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)  # pyrefly: ignore[unsupported-operation]
  up = np.cross(right, forward)
  rot = np.stack([right, up, -forward], axis=1).astype(np.float32)
  cam = wp.transformf(wp.vec3f(*eye_np), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
  return wp.array([cam] * count, dtype=wp.transformf, device=device)


_OUTPUTS = ("color", "depth", "depth_forward", "shape_index", "normal", "albedo", "color_hdr")


def _render(cls, model, state, cams, rays, world_indices=None, **kwargs) -> dict[str, np.ndarray]:
  views = cams.shape[0]
  height, width = rays.shape[:2]
  with wp.ScopedDevice(model.device):
    sensor = cls(model, **kwargs)
    sensor.create_default_light()
    images = {name: getattr(sensor, f"allocate_image_{name}")(views, width, height) for name in _OUTPUTS}
    model.bvh_build_shapes(state)
    sensor.update(
      state,
      cams,
      rays,
      color_image=images["color"],
      depth_image=images["depth"],
      forward_depth_image=images["depth_forward"],
      shape_index_image=images["shape_index"],
      normal_image=images["normal"],
      albedo_image=images["albedo"],
      hdr_color_image=images["color_hdr"],
      world_indices=world_indices,
    )
  return {name: image.numpy().copy() for name, image in images.items()}


def _bytes(packed: np.ndarray) -> np.ndarray:
  return np.stack([np.right_shift(packed, s) & 0xFF for s in (0, 8, 16, 24)], axis=-1).astype(np.int32)


def _assert_geometry_matches(ours: dict[str, np.ndarray], ref: dict[str, np.ndarray], min_same: float = 0.995) -> np.ndarray:
  same = ours["shape_index"] == ref["shape_index"]
  assert same.mean() >= min_same
  hit = same & (ref["shape_index"] != 0xFFFFFFFF)
  assert hit.mean() > 0.3
  for name in ("depth", "depth_forward"):
    np.testing.assert_allclose(ours[name][hit], ref[name][hit], rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(ours[name][~hit & same], ref[name][~hit & same])
  np.testing.assert_allclose(ours["normal"][hit], ref["normal"][hit], atol=1e-2)
  # Albedo: the shape colour in sRGB bytes (rounding of the colour conversions differs).
  assert np.abs(_bytes(ours["albedo"])[hit] - _bytes(ref["albedo"])[hit]).max() <= 1
  np.testing.assert_array_equal(ours["albedo"][~hit & same], ref["albedo"][~hit & same])
  return hit


@pytest.fixture(scope="module", params=_DEVICES)
def scene(request):
  return _scene(device=request.param)


def test_pinhole_parity_with_newton(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  ours = _render(SensorCamera, model, state, cams, rays)
  ref = _render(NewtonSensorCamera, model, state, cams, rays)
  hit = _assert_geometry_matches(ours, ref)
  # Both worlds and every shape type are visible; colour only sanity-checked (shading differs).
  assert len(np.unique(ours["shape_index"][hit])) >= 9
  box = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.BOX))[0])
  r, g, b, a = _bytes(ours["color"])[ours["shape_index"] == box].mean(axis=0)
  assert r > 2.0 * g and r > 2.0 * b and a == 255
  miss = ref["shape_index"] == 0xFFFFFFFF
  np.testing.assert_array_equal(ours["color"][miss], 0)  # ClearData().clear_color
  np.testing.assert_array_equal(ours["color_hdr"][miss], 0.0)
  assert np.all(ours["color_hdr"][hit] >= 0.0) and ours["color_hdr"][hit].max() > 0.05


def test_fisheye_parity_with_newton(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_fisheye_opencv(
    _RES, _RES, fx=14.0, fy=14.0, cx=_RES / 2, cy=_RES / 2, k1=0.05, max_fov=math.radians(170.0), device=model.device
  )
  ours = _render(SensorCamera, model, state, cams, rays)
  ref = _render(NewtonSensorCamera, model, state, cams, rays)
  _assert_geometry_matches(ours, ref, min_same=0.99)


def test_views_follow_world_indices(scene):
  model, state, _ = scene
  # Stereo pair of world 1, then world 0, a disabled view and an out-of-range one.
  cams = wp.array(
    [*_look_at([-0.1, -3.0, 1.2], 1).numpy(), *_look_at([0.1, -3.0, 1.2], 1).numpy(), *_look_at([0.0, -3.0, 1.2], 3).numpy()],
    dtype=wp.transformf,
    device=model.device,
  )
  flags = [1, 1, 0, int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), 7]
  world_indices = wp.array(flags, dtype=wp.int32, device=model.device)
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  ours = _render(SensorCamera, model, state, cams, rays, world_indices=world_indices)
  ref = _render(NewtonSensorCamera, model, state, cams, rays, world_indices=world_indices)
  _assert_geometry_matches(ours, ref)
  # The two eyes see the same world from 20 cm apart; disabled / invalid views are cleared.
  assert not np.array_equal(ours["shape_index"][0], ours["shape_index"][1])
  for view in (3, 4):
    assert np.all(ours["shape_index"][view] == 0xFFFFFFFF) and np.all(ours["depth"][view] == 0.0)
    assert np.all(ours["color"][view] == 0)


def test_clear_data_and_linear_color_space(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  clear = SensorCamera.ClearData(
    clear_color=0xFF336699, clear_depth=-1.0, clear_shape_index=77, clear_normal=(0.0, 1.0, 0.0), clear_albedo=0x80112233
  )
  config = SensorCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)
  results = {}
  for cls in (SensorCamera, NewtonSensorCamera):
    with wp.ScopedDevice(model.device):
      sensor = cls(model, default_clear_data=clear, default_render_config=config)
      color = sensor.allocate_image_color(2, _RES, _RES)
      albedo = sensor.allocate_image_albedo(2, _RES, _RES)
      depth = sensor.allocate_image_depth(2, _RES, _RES)
      index = sensor.allocate_image_shape_index(2, _RES, _RES)
      normal = sensor.allocate_image_normal(2, _RES, _RES)
      model.bvh_build_shapes(state)
      sensor.update(
        state,
        cams,
        rays,
        color_image=color,
        albedo_image=albedo,
        depth_image=depth,
        shape_index_image=index,
        normal_image=normal,
      )
    results[cls] = [a.numpy().copy() for a in (color, albedo, depth, index, normal)]
  ours, ref = results[SensorCamera], results[NewtonSensorCamera]
  miss = ref[3] == 77
  assert miss.any()
  np.testing.assert_array_equal(ours[3] == 77, miss)
  for a, b in zip(ours, ref, strict=True):
    np.testing.assert_array_equal(a[miss], b[miss])
  hit = ~miss & (ours[3] == ref[3])
  assert np.abs(_bytes(ours[1])[hit] - _bytes(ref[1])[hit]).max() <= 1  # linear albedo bytes


def test_shadows_follow_render_config(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    sensor.create_default_light(direction=wp.vec3f(0.0, 0.6, -0.8))
    colors = []
    for shadows in (False, True):
      color = sensor.allocate_image_color(2, _RES, _RES)
      config = SensorCamera.RenderConfig(enable_shadows=shadows)
      sensor.update(state, cams, rays, color_image=color, render_config=config)
      colors.append(_bytes(color.numpy())[..., :3].sum())
  assert colors[1] < colors[0]


def test_render_options_and_lights():
  model, state, cams = _scene(nworld=1)
  with pytest.raises(TypeError, match="cam_res"):
    SensorCamera(model, cam_res=(8, 8))
  rays = SensorCamera.compute_camera_rays_pinhole(16, 16, camera_fov=_FOV, device="cpu")
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model, lights=[newton_pbr.Light("directional", dir=(0.0, 0.0, -1.0))], tier=1, path_samples=2)
    color = sensor.allocate_image_color(1, 16, 16)
    sensor.update(state, cams, rays, color_image=color)
  assert _bytes(color.numpy())[..., :3].max() > 50


def test_deformables_warn_once():
  builder = newton.ModelBuilder()
  builder.add_shape_sphere(-1, radius=0.3)
  builder.add_particle((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0, radius=0.05)
  with wp.ScopedDevice("cpu"):
    model = builder.finalize(device="cpu")
    with pytest.warns(UserWarning, match="rigid shapes only"):
      SensorCamera(model)


def test_checkerboard_material(scene):
  model, state, cams = scene
  plane = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))[0])
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    sensor.assign_checkerboard_material(shape_indices=[plane], resolution=8, checker_size=1)
    albedo = sensor.allocate_image_albedo(2, _RES, _RES)
    index = sensor.allocate_image_shape_index(2, _RES, _RES)
    for textures in (False, True):
      config = SensorCamera.RenderConfig(enable_textures=textures)
      sensor.update(state, cams, rays, albedo_image=albedo, shape_index_image=index, render_config=config)
      floor = _bytes(albedo.numpy())[index.numpy() == plane][:, 0]
      # Two greys (sRGB 0x80 and 0xBF times the shape colour) with textures, one colour without.
      assert (floor.max() > 1.3 * floor.min()) == textures
  with pytest.raises(ValueError, match="out-of-range"):
    sensor.assign_checkerboard_material(shape_indices=[model.shape_count])
