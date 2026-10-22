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

"""Shared fixtures: build device model/data and render a scene from inline XML."""

import dataclasses

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr


@dataclasses.dataclass
class Scene:
  mjm: mujoco.MjModel
  mjd: mujoco.MjData
  m: mjw.Model
  d: mjw.Data
  rc: mjpbr.PbrRenderContext

  def render(self) -> "Scene":
    mjw.refit_bvh(self.m, self.d, self.rc.base)
    mjpbr.render(self.m, self.d, self.rc)
    return self

  def rgb(self, camera: int = 0) -> np.ndarray:
    w, h = self.rc.base.cam_res.numpy()[camera]
    out = wp.zeros((self.d.nworld, h, w), dtype=wp.vec3)
    mjpbr.get_rgb(self.rc, camera, out)
    return out.numpy()

  def hdr(self, camera: int = 0) -> np.ndarray:
    w, h = self.rc.base.cam_res.numpy()[camera]
    out = wp.zeros((self.d.nworld, h, w), dtype=wp.vec3)
    mjpbr.get_hdr(self.rc, camera, out)
    return out.numpy()


def make_scene(xml: str, nworld: int = 1, cam_res=(32, 32), **kwargs) -> Scene:
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  kwargs.setdefault("render_rgb", True)
  rc = mjpbr.create_render_context(mjm, nworld=nworld, cam_res=cam_res, **kwargs)
  return Scene(mjm, mjd, m, d, rc)


@pytest.fixture
def scene():
  return make_scene
