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

"""Newton's ViewerRTX (NVIDIA Omniverse RTX through `ovrtx`) on a MuJoCo scene, headless.

Requires an RTX GPU and `pip install "newton[rtx]"` (ovrtx and ovstage from pypi.nvidia.com).
The viewer renders one camera at a time (it is not a batched sensor), with its own lighting
preset ("studio") instead of the scene's lights and HDRI. The MuJoCo camera "cam" is mapped to
the viewer's position / pitch / yaw (Z-up, no roll) and field of view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from references.newton_camera import newton_scene

if TYPE_CHECKING:
  from _common import RGB


class NewtonRtxCamera:
  def __init__(self, spec: mujoco.MjSpec, mjm: mujoco.MjModel, res: int, environment: str = "studio"):
    from newton.viewer import ViewerRTX

    self.model, self.state, _, fovy = newton_scene(spec, mjm, 1)
    self.viewer = ViewerRTX(width=res, height=res, headless=True, environment=environment, async_rendering=False)
    self.viewer.set_model(self.model)
    self.viewer.camera.fov = fovy
    self.mjm = mjm
    self.mjd = mujoco.MjData(mjm)
    self.cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    self.time = 0.0
    self.update_camera()

  def update_camera(self):
    """Copies the pose of the MuJoCo camera "cam" (after mj_forward) to the viewer."""
    mujoco.mj_forward(self.mjm, self.mjd)
    forward = -self.mjd.cam_xmat[self.cam].reshape(3, 3)[:, 2]
    pitch = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(forward[1], forward[0])))
    self.viewer.set_camera(self.mjd.cam_xpos[self.cam].copy(), pitch=pitch, yaw=yaw)

  def render(self):
    self.time += 1.0 / 60.0
    self.viewer.begin_frame(self.time)
    self.viewer.log_state(self.state)
    self.viewer.end_frame()
    self.frame = self.viewer.get_frame()

  def close(self):
    """Releases the RTX renderer (one ViewerRTX at a time per process)."""
    self.viewer.close()

  def image(self) -> RGB[int, int]:
    """Last frame as float RGB in [0, 1], (res, res, 3)."""
    return self.frame.numpy()[..., :3].astype(np.float32) / 255.0
