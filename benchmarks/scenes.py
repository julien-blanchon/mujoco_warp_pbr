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
import numpy as np

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


def _bake_pose(spec: mujoco.MjSpec, keyframe: str | None = None) -> mujoco.MjSpec:
  """Freezes the model at a keyframe (or qpos0): bodies get their posed local frames and every
  joint, actuator, sensor, tendon, equality and keyframe is removed, so any consumer (MuJoCo,
  Newton's MJCF importer, Mitsuba) sees the posed, static scene."""
  mjm = spec.compile()
  mjd = mujoco.MjData(mjm)
  if keyframe is not None:
    mujoco.mj_resetDataKeyframe(mjm, mjd, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, keyframe))
  mujoco.mj_forward(mjm, mjd)
  for body in spec.bodies[1:]:
    b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, body.name)
    parent = mjm.body_parentid[b]
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, mjd.xquat[parent])
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, mjd.xpos[b] - mjd.xpos[parent], inv)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, inv, mjd.xquat[b])
    body.pos, body.quat = pos, quat
  for elements in (spec.actuators, spec.sensors, spec.tendons, spec.equalities, spec.keys, spec.joints):
    for element in list(elements):
      spec.delete(element)
  return spec


def _stage(spec: mujoco.MjSpec, camera_pos, camera_xyaxes, fovy: float = 45.0, sun: bool = True):
  """Adds a floor, the camera and (optionally) the standard sun to a robot / asset spec."""
  spec.visual.headlight.ambient = [0.2] * 3
  spec.visual.headlight.diffuse = [0.3] * 3
  spec.visual.headlight.specular = [0.3] * 3
  spec.add_material(name="stage_floor", rgba=[0.5, 0.5, 0.5, 1.0], roughness=0.9)
  world = spec.worldbody
  if sun:
    world.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[-0.4, 0.6, -0.7], castshadow=True)
  world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], material="stage_floor")
  world.add_camera(name="cam", pos=list(camera_pos), xyaxes=list(camera_xyaxes), fovy=fovy)
  return spec


def menagerie_anymal() -> mujoco.MjSpec:
  """MuJoCo Menagerie ANYmal C (textured) standing on a floor (requires tools/fetch_assets.py)."""
  path = asset_dir() / "menagerie" / "anybotics_anymal_c" / "anymal_c.xml"
  if not path.exists():
    raise FileNotFoundError(f"{path} missing: run tools/fetch_assets.py")
  spec = _bake_pose(mujoco.MjSpec.from_file(str(path)))
  return _stage(spec, camera_pos=[1.3, -1.5, 0.9], camera_xyaxes=[0.76, 0.65, 0, -0.25, 0.29, 0.92])


def franka_showroom() -> mujoco.MjSpec:
  """Menagerie Franka Panda (home pose) lit only by an authored rig, no environment map: a warm
  key spot with a soft bulb, a cool fill point light and a rim light from behind."""
  path = asset_dir() / "menagerie" / "franka_emika_panda" / "panda.xml"
  if not path.exists():
    raise FileNotFoundError(f"{path} missing: run tools/fetch_assets.py")
  spec = _bake_pose(mujoco.MjSpec.from_file(str(path)), keyframe="home")
  spec = _stage(spec, camera_pos=[1.6, -1.2, 0.95], camera_xyaxes=[0.6, 0.8, 0, -0.3, 0.22, 0.93], sun=False)
  spec.visual.headlight.active = 0
  world = spec.worldbody
  world.add_light(
    name="key",
    type=mujoco.mjtLightType.mjLIGHT_SPOT,
    pos=[1.2, -0.8, 1.8],
    dir=[-0.55, 0.35, -0.75],
    diffuse=[1.0, 0.92, 0.8],
    cutoff=40,
    bulbradius=0.08,
    castshadow=True,
  )
  world.add_light(
    name="fill",
    type=mujoco.mjtLightType.mjLIGHT_POINT,
    pos=[-0.6, -1.4, 0.9],
    diffuse=[0.35, 0.42, 0.55],
    bulbradius=0.15,
    castshadow=False,
  )
  world.add_light(
    name="rim",
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    dir=[0.3, -0.9, -0.3],
    diffuse=[0.5, 0.55, 0.6],
    castshadow=True,
  )
  return spec


def isaac(scene: str, camera: str = "cam", stage: dict | None = None) -> Callable[[], mujoco.MjSpec]:
  """A converted NVIDIA Isaac Sim / IsaacLab USD scene (requires `fetch_assets.py --only isaac`;
  local use only: the assets must not be redistributed). Scenes keep their converted UsdLux
  lights; `camera` names the converted USD camera used as "cam". With `stage` (kwargs of _stage)
  the scene gets the standard floor, sun and camera instead."""

  def make() -> mujoco.MjSpec:
    path = asset_dir() / "mjcf" / scene / f"{scene}.xml"
    if not path.exists():
      raise FileNotFoundError(f"{path} missing: run tools/fetch_assets.py --only isaac")
    spec = mujoco.MjSpec.from_file(str(path))
    if stage is not None:
      return _stage(spec, **stage)
    cam = spec.camera(camera)
    if cam is None:
      raise ValueError(f"{path} has no camera {camera!r}")
    cam.name = "cam"
    return spec

  return make


SCENES = {
  "material_grid": ("material", material_grid),
  "primitives": ("robot", primitives),
  "damaged_helmet": ("asset", khronos("DamagedHelmet")),
  "flight_helmet": ("asset", khronos("FlightHelmet")),
  "scifi_helmet": ("asset", khronos("SciFiHelmet")),
  "a_beautiful_game": ("asset", khronos("ABeautifulGame", extent=0.65)),
  "corset": ("asset", khronos("Corset")),
  "lantern": ("asset", khronos("Lantern")),
  "anymal_c": ("robot", menagerie_anymal),
  "franka_showroom": ("robot", franka_showroom),
  # NVIDIA Isaac USD scenes converted by tools/usd_to_mjcf.py (lights and cameras from USD).
  "isaac_renderer_gallery": ("asset", isaac("IsaacRendererGallery", camera="Camera")),
  "isaac_franka_cabinet": ("robot", isaac("IsaacFrankaCabinet")),
  "isaac_seattle_lab_table": (
    "asset",
    isaac(
      "IsaacSeattleLabTable", stage={"camera_pos": [2.4, -1.5, 2.1], "camera_xyaxes": [0.679, 0.734, 0, -0.296, 0.274, 0.915]}
    ),
  ),
}

# Scenes lit by their own lights only: benchmarks pass no environment map for them.
LIGHTS_ONLY = {"franka_showroom", "isaac_renderer_gallery", "isaac_franka_cabinet"}
