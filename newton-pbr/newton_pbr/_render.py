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

"""Shared by the sensors and the viewer: Newton poses -> MuJoCo Warp data of a scene mapping."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp

from newton_pbr._scene import SceneMapping
from newton_pbr._scene import _required


@wp.kernel(enable_backward=False)
def _sync_poses(
  body_q: wp.array1d[wp.transformf],
  shape_body: wp.array1d[wp.int32],
  shape_transform: wp.array1d[wp.transformf],
  shape_world: wp.array1d[wp.int32],
  geom_shape: wp.array2d[wp.int32],
  offset_pos: wp.array1d[wp.vec3],
  offset_quat: wp.array1d[wp.quat],
  view_world: wp.array1d[wp.int32],
  has_view_world: bool,
  world_offsets: wp.array1d[wp.vec3],
  has_world_offsets: bool,
  camera_transforms: wp.array1d[wp.transformf],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  cam_xpos: wp.array2d[wp.vec3],
  cam_xmat: wp.array2d[wp.mat33],
):
  """Geom poses of render world (view) v from Newton's body transforms, and its camera pose.

  View v shows Newton world view_world[v] (v without a mapping; world 0 for a disabled view,
  whose outputs are not used). With world offsets, a shape of world w is moved by offset w.
  """
  view, geom = wp.tid()
  world = view
  if has_view_world:
    world = view_world[view]
  if world < 0 or world >= geom_shape.shape[0]:
    world = 0
  shape = geom_shape[world, geom]
  x = shape_transform[shape]
  body = shape_body[shape]
  if body >= 0:
    x = wp.transform_multiply(body_q[body], x)
  x = wp.transform_multiply(x, wp.transform(offset_pos[geom], offset_quat[geom]))
  pos = wp.transform_get_translation(x)
  if has_world_offsets:
    w = shape_world[shape]
    if w >= 0 and w < world_offsets.shape[0]:
      pos = pos + world_offsets[w]
  geom_xpos[view, geom] = pos
  geom_xmat[view, geom] = wp.quat_to_matrix(wp.transform_get_rotation(x))
  if geom == 0:
    # Newton and MuJoCo cameras both look along -Z with +Y up.
    c = camera_transforms[view]
    cam_xpos[view, 0] = wp.transform_get_translation(c)
    cam_xmat[view, 0] = wp.quat_to_matrix(wp.transform_get_rotation(c))


class ScenePoses:
  """Writes the poses of a scene mapping's geoms (and its camera) into MuJoCo Warp data."""

  def __init__(self, model: newton.Model, scene: SceneMapping):
    device = model.device
    self.model = model
    self.ngeom = int(scene.mjm.ngeom)
    self.geom_shape = wp.array(scene.geom_shape, dtype=wp.int32, device=device)
    self.offset_pos = wp.array(scene.geom_offset_pos.astype(np.float32), dtype=wp.vec3, device=device)
    # MuJoCo quaternions are (w, x, y, z), Warp's (x, y, z, w).
    q = scene.geom_offset_quat[:, [1, 2, 3, 0]].astype(np.float32)
    self.offset_quat = wp.array(q, dtype=wp.quat, device=device)
    self._no_world = wp.zeros(1, dtype=wp.int32, device=device)
    self._no_offsets = wp.zeros(1, dtype=wp.vec3, device=device)
    self._no_bodies = wp.zeros(1, dtype=wp.transformf, device=device)

  def sync(
    self,
    state: newton.State,
    camera_transforms: wp.array1d[wp.transformf],
    d: mjw.Data,
    view_world: wp.array1d[wp.int32] | None = None,
    world_offsets: wp.array1d[wp.vec3] | None = None,
  ) -> None:
    """Poses of the d.nworld render worlds (views); camera_transforms has one per view."""
    if self.ngeom == 0:
      return
    model = self.model
    wp.launch(
      _sync_poses,
      dim=(d.nworld, self.ngeom),
      inputs=[
        # A model without bodies has no body_q (all shapes are static).
        state.body_q if state.body_q is not None else self._no_bodies,
        _required(model.shape_body, "shape_body"),
        _required(model.shape_transform, "shape_transform"),
        _required(model.shape_world, "shape_world"),
        self.geom_shape,
        self.offset_pos,
        self.offset_quat,
        view_world if view_world is not None else self._no_world,
        view_world is not None,
        world_offsets if world_offsets is not None else self._no_offsets,
        world_offsets is not None,
        camera_transforms,
      ],
      outputs=[d.geom_xpos, d.geom_xmat, d.cam_xpos, d.cam_xmat],
      device=model.device,
    )


def put_data(mjm: mujoco.MjModel, nworld: int, device: wp.Device) -> mjw.Data:
  """MuJoCo Warp data of nworld worlds (only the geom and camera poses are used)."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with wp.ScopedDevice(device):
    return mjw.put_data(mjm, mjd, nworld=nworld)


def put_scene(scene: SceneMapping, nworld: int, device: wp.Device) -> tuple[mjw.Model, mjw.Data]:
  """MuJoCo Warp model and data (nworld worlds) of a scene mapping on `device`."""
  with wp.ScopedDevice(device):
    m = mjw.put_model(scene.mjm)
  return m, put_data(scene.mjm, nworld, device)
