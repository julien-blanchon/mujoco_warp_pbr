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

"""SensorPbrCamera against Newton's own SensorCamera on the same scene."""

from typing import Any

import newton
import numpy as np
import pytest
import warp as wp
from newton.sensors import SensorCamera

import newton_pbr

_RES = 48
_FOVY = 50.0


def _scene(nworld: int = 2) -> tuple[newton.Model, newton.State, wp.array]:
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
  world.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, 0.45), wp.quat_identity()), radius=0.2)
  arm = world.add_body(xform=wp.transform((0.7, 0.0, 0.4), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.6)))
  world.add_shape_capsule(arm, radius=0.1, half_height=0.25)
  # An off-centre tetrahedron: MuJoCo re-centres mesh vertices, the sensor must undo it.
  verts = np.array([[0.4, 0.4, 0.0], [0.8, 0.4, 0.0], [0.4, 0.8, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
  faces = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32)
  world.add_shape_mesh(-1, mesh=newton.Mesh(verts, faces), xform=wp.transform((-1.0, -0.6, 0.0), wp.quat_identity()))
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, nworld)
  model = scene.finalize()
  state = model.state()
  # World 1 is shifted and turned so the two views differ.
  q = state.body_q.numpy()
  per_world = len(q) // nworld
  q[per_world:, 1] += 0.3
  state.body_q.assign(q)
  # Camera at (0, -3, 1.2) looking at the origin: -Z forward, +Y up.
  eye = np.array([0.0, -3.0, 1.2])
  forward = -eye / np.linalg.norm(eye)
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  rot = np.stack([right, up, -forward], axis=1).astype(np.float32)
  cam = wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
  cams = wp.array([cam] * nworld, dtype=wp.transformf)
  return model, state, cams


def _newton_reference(model, state, cams):
  sensor = SensorCamera(model)
  sensor.create_default_light()
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=np.radians(_FOVY))
  index = sensor.allocate_image_shape_index(cams.shape[0], _RES, _RES)
  depth = sensor.allocate_image_depth_forward(cams.shape[0], _RES, _RES)
  model.bvh_build_shapes(state)
  sensor.update(state, cams, rays, shape_index_image=index, forward_depth_image=depth)
  return index.numpy(), depth.numpy()


@pytest.fixture(scope="module")
def rendered():
  model, state, cams = _scene()
  camera = newton_pbr.SensorPbrCamera(model, width=_RES, height=_RES, fovy=_FOVY)
  color = camera.allocate_image_color()
  index = camera.allocate_image_shape_index()
  depth = camera.allocate_image_depth()
  forward = camera.allocate_image_depth()
  camera.update(state, cams, color_image=color, shape_index_image=index, depth_image=depth, forward_depth_image=forward)
  return model, state, cams, color.numpy(), index.numpy(), depth.numpy(), forward.numpy()


def test_shape_index_and_depth_match_newton(rendered):
  model, state, cams, _, index, _, forward = rendered
  ref_index, ref_depth = _newton_reference(model, state, cams)
  # Shape ids agree everywhere except on silhouettes (ray generation differs slightly).
  edge = np.zeros(ref_index.shape, dtype=bool)
  for shift in ((0, 1), (1, 0), (0, -1), (-1, 0)):
    edge |= np.roll(ref_index, shift, axis=(1, 2)) != ref_index
  assert not np.any((index != ref_index) & ~edge)
  assert np.mean(index != ref_index) < 0.05
  same = (index == ref_index) & (ref_depth > 0.0)
  np.testing.assert_allclose(forward[same], ref_depth[same], rtol=2e-3, atol=2e-3)
  # Both worlds and every shape type are visible.
  assert len(np.unique(index[index != 0xFFFFFFFF])) >= 9


def test_ray_depth_exceeds_forward_depth(rendered):
  *_, depth, forward = rendered
  hit = forward > 0.0
  assert np.all(depth[hit] >= forward[hit] - 1e-5)
  assert np.all(depth[~hit] < 0.0)


def test_shape_colours(rendered):
  model, _, _, color, index, _, _ = rendered
  rgba = np.stack([(color >> s) & 0xFF for s in (0, 8, 16, 24)], axis=-1)
  box = np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.BOX))[0]
  r, g, b = rgba[index == box][:, :3].mean(axis=0)
  # The red box (sRGB 0.9, 0.1, 0.1) stays red after lighting and tone mapping.
  assert r > 2.0 * g and r > 2.0 * b


def test_light_type_is_validated():
  with pytest.raises(ValueError, match="Light.type"):
    kind: Any = "area"
    newton_pbr.Light(kind)


def test_reserved_render_options_raise():
  model, _, _ = _scene(nworld=1)
  with pytest.raises(TypeError, match="render_rgb"):
    newton_pbr.SensorPbrCamera(model, width=8, height=8, render_rgb=False)


def test_output_images_are_validated(rendered):
  model, state, cams, *_ = rendered
  camera = newton_pbr.SensorPbrCamera(model, width=16, height=16)
  wrong: Any = wp.zeros((cams.shape[0], 8, 16), dtype=wp.uint32, device=model.device)
  with pytest.raises(ValueError, match="color_image"):
    camera.update(state, cams, color_image=wrong)
  with pytest.raises(ValueError, match="depth_image"):
    index: Any = camera.allocate_image_shape_index()
    camera.update(state, cams, depth_image=index)


def test_worlds_must_match():
  def world(radius: float) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=radius)
    return builder

  scene = newton.ModelBuilder()
  scene.add_world(world(0.2))
  scene.add_world(world(0.3))
  with pytest.raises(ValueError, match="same visible shapes"):
    newton_pbr.SensorPbrCamera(scene.finalize(), width=8, height=8)


def test_shared_meshes_become_one_mujoco_mesh():
  verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  faces = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32)
  mesh = newton.Mesh(verts, faces)
  builder = newton.ModelBuilder()
  for x in (0.0, 1.5):
    builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
  builder.add_shape_mesh(-1, mesh=mesh, scale=(2.0, 2.0, 2.0))
  scene = newton_pbr.build_scene(builder.finalize(), None, 45.0)
  assert scene.mjm.ngeom == 3
  assert scene.mjm.nmesh == 2  # one per (mesh, scale)
