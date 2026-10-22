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

"""Benchmark scenes.

Each scene is a function returning a mujoco.MjSpec with one camera named "cam" (compile it for
MuJoCo; `to_xml()` feeds other renderers). Scenes built from downloaded assets read them from
$MJPBR_ASSET_DIR (see tools/fetch_assets.py).

Groups: "material" (correctness, group 1), "asset" (asset-heavy PBR, group 3) and "robot"
(robotics throughput, group 2).
"""

import os
from pathlib import Path
from typing import Callable

import mujoco

_LIGHT = '<light name="sun" type="directional" dir="-0.4 0.6 -0.7" castshadow="true"/>'


def asset_dir() -> Path:
  return Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "mujoco_warp_pbr"))


def material_grid(n: int = 7) -> mujoco.MjSpec:
  """n × n spheres: metallic increases bottom to top, roughness left to right."""
  mats, geoms = [], []
  spacing = 0.22
  for i in range(n):
    for j in range(n):
      metallic, roughness = i / (n - 1), j / (n - 1)
      mats.append(f'<material name="m{i}_{j}" rgba="0.95 0.64 0.54 1" metallic="{metallic}" roughness="{roughness}"/>')
      x, z = (j - (n - 1) / 2) * spacing, 0.12 + i * spacing
      geoms.append(f'<geom type="sphere" size="0.09" pos="{x:.3f} 0 {z:.3f}" material="m{i}_{j}"/>')
  height = 0.12 + (n - 1) * spacing
  return mujoco.MjSpec.from_string(f"""
<mujoco model="material_grid">
  <visual><headlight ambient="0.2 0.2 0.2" diffuse="0.3 0.3 0.3" specular="0.3 0.3 0.3"/></visual>
  <asset>
    <material name="floor" rgba="0.5 0.5 0.5 1" roughness="0.9"/>
    {"".join(mats)}
  </asset>
  <worldbody>
    {_LIGHT}
    <geom type="plane" size="3 3 0.1" material="floor"/>
    {"".join(geoms)}
    <camera name="cam" pos="0 {-0.55 - 1.9 * height:.3f} {0.5 * height + 0.06:.3f}" xyaxes="1 0 0 0 0 1" fovy="45"/>
  </worldbody>
</mujoco>
""")


def primitives() -> mujoco.MjSpec:
  """MuJoCo Warp's render benchmark layout: 50 primitives on a table, flat materials."""
  bodies = []
  for kind, size, z, rgba in [
    ("sphere", "0.05", 0.8, "1 0 0 1"),
    ("capsule", "0.04 0.05", 0.95, "0 1 0 1"),
  ]:
    for i in range(5):
      for j in range(5):
        x, y = 0.6 + 0.1 * i, -0.4 + 0.2 * j
        bodies.append(f'<geom type="{kind}" size="{size}" pos="{x:.2f} {y:.2f} {z}" rgba="{rgba}"/>')
  return mujoco.MjSpec.from_string(f"""
<mujoco model="primitives">
  <visual><map znear="0.01" zfar="200"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 .05" material="grid"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom type="box" pos="0.8 0 0.5" size="0.5 1.0 0.01"/>
    {"".join(bodies)}
    <camera name="cam" pos="3 0 2" xyaxes="0 1 0 -0.6 0 1"/>
  </worldbody>
</mujoco>
""")


def khronos(model: str, extent: float = 1.0) -> Callable[[], mujoco.MjSpec]:
  """A converted Khronos sample model on a floor (requires tools/fetch_assets.py)."""

  def scene() -> mujoco.MjSpec:
    path = asset_dir() / "mjcf" / model / f"{model}.xml"
    if not path.exists():
      raise FileNotFoundError(f"{path} missing: run tools/fetch_assets.py")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.visual.headlight.ambient = [0.2] * 3
    spec.visual.headlight.diffuse = [0.3] * 3
    spec.visual.headlight.specular = [0.3] * 3
    spec.add_material(name="floor", rgba=[0.5, 0.5, 0.5, 1.0], roughness=0.9)
    world = spec.worldbody
    world.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[-0.4, 0.6, -0.7], castshadow=True)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="floor")
    world.add_camera(name="cam", pos=[0, -1.7 * extent, 0.55 * extent], xyaxes=[1, 0, 0, 0, 0.15, 1], fovy=45)
    return spec

  return scene


SCENES = {
  "material_grid": ("material", material_grid),
  "primitives": ("robot", primitives),
  "damaged_helmet": ("asset", khronos("DamagedHelmet")),
  "flight_helmet": ("asset", khronos("FlightHelmet")),
  "scifi_helmet": ("asset", khronos("SciFiHelmet")),
}
