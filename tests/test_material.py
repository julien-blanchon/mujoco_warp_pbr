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

"""Tests for resolving unset PBR material fields."""

import mujoco
import numpy as np

from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src.types import UnsetMaterial

_XML = """
  <mujoco>
    <asset>
      <material name="authored" metallic="0.3" roughness="0.6" specular="0.7"/>
      <material name="legacy" shininess="0.8" rgba="0.5 0.5 0.5 1"/>
      <material name="white" rgba="0.9 0.9 0.9 1"/>
    </asset>
    <worldbody>
      <geom name="authored" type="sphere" size="0.1" material="authored"/>
      <geom name="legacy" type="sphere" size="0.1" material="legacy"/>
      <geom name="white" type="sphere" size="0.1" material="white"/>
      <geom name="nomat" type="sphere" size="0.1" rgba="0.1 0.1 0.1 1"/>
      <geom name="floor" type="plane" size="1 1 0.1" material="legacy"/>
    </worldbody>
  </mujoco>
"""


def _resolve(**kwargs):
  mjm = mujoco.MjModel.from_xml_string(_XML)
  metallic, roughness, reflectance = material.resolve_geom_materials(mjm, **kwargs)
  names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(mjm.ngeom)]
  return {n: (metallic[i], roughness[i], reflectance[i]) for i, n in enumerate(names)}


def test_gltf_policy():
  r = _resolve()
  np.testing.assert_allclose(r["authored"], (0.3, 0.6, 0.7), rtol=1e-6)
  # Unset: metallic 0, roughness 1 - shininess, reflectance = material specular (default 0.5).
  np.testing.assert_allclose(r["legacy"], (0.0, 0.2, 0.5), rtol=1e-6)
  np.testing.assert_allclose(r["nomat"], (0.0, 0.5, 0.5), rtol=1e-6)


def test_default_material_only_fills_unset_fields():
  r = _resolve(default_material={"metallic": 1.0, "roughness": 0.1})
  np.testing.assert_allclose(r["authored"][:2], (0.3, 0.6), rtol=1e-6)
  np.testing.assert_allclose(r["legacy"][:2], (1.0, 0.1), rtol=1e-6)


def test_heuristic_policy():
  r = _resolve(unset_material=UnsetMaterial.HEURISTIC)
  np.testing.assert_allclose(r["authored"], (0.3, 0.6, 0.7), rtol=1e-6)
  # Mid grey -> a touch metallic; roughness nudged by the non-default shininess 0.8.
  np.testing.assert_allclose(r["legacy"], (0.6, 0.40 * (1.3 - 0.8), 0.7), rtol=1e-5)
  # Bright neutral -> satin plastic.
  np.testing.assert_allclose(r["white"], (0.0, 0.38, 0.65), rtol=1e-6)
  # Dark geom without material.
  np.testing.assert_allclose(r["nomat"], (0.4, 0.45, 0.35), rtol=1e-6)
  # Ground planes are matte.
  np.testing.assert_allclose(r["floor"], (0.0, 1.0, 0.0), rtol=1e-6)
