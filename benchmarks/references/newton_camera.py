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

"""Newton's Warp ray-traced camera sensor on a MuJoCo scene, as a speed reference.

Uses the settings of Newton's own camera benchmark (SAH BVH, 8x8 tiled order, block 64,
shadows and textures on). Shading is Newton's (Lambert + hemispheric ambient), not PBR.
"""

import os

import mujoco
import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera


class NewtonCamera:
  def __init__(self, spec: mujoco.MjSpec, mjm: mujoco.MjModel, nworld: int, res: int, depth: bool = False):
    base_dir = spec.modelfiledir or os.getcwd()
    world = newton.ModelBuilder()
    world.add_mjcf(
      spec.to_xml(), path_resolver=lambda base, path: path if os.path.isabs(path) else os.path.join(base_dir, path)
    )
    world.shape_collision_filter_pairs = []
    scene = newton.ModelBuilder()
    scene.replicate(world, nworld)
    self.model = scene.finalize()
    self.state = self.model.state()
    newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    rot = mjd.cam_xmat[cam].reshape(3, 3).astype(np.float32)
    transform = wp.transformf(wp.vec3f(*mjd.cam_xpos[cam]), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
    row = np.array([transform[i] for i in range(7)], dtype=np.float32)
    self.cameras = wp.array(np.tile(row, (nworld, 1)), dtype=wp.transformf)

    self.sensor = SensorCamera(self.model)
    self.sensor.create_default_light(enable_shadows=True)
    config = self.sensor.default_render_config
    config.render_order = SensorCamera.RenderOrder.TILED
    config.tile_width = 8
    config.tile_height = 8
    config.enable_shadows = True
    config.enable_textures = True
    config.block_dim = 64
    self.rays = SensorCamera.compute_camera_rays_pinhole(res, res, camera_fov=np.radians(float(mjm.cam_fovy[cam])))
    self.color = self.sensor.allocate_image_color(nworld, res, res)
    self.depth = self.sensor.allocate_image_depth(nworld, res, res) if depth else None
    self.model.bvh_build_shapes(self.state, bvh_constructor="sah")

  def render(self):
    self.sensor.update(
      self.state, self.cameras, self.rays, color_image=self.color, depth_image=self.depth, sync_deformables=False
    )
