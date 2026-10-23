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

"""Mesh BVH builders and frames: no leftover BVHs, same images."""

import numpy as np
import pytest


def _l_shape() -> tuple[str, str]:
  """(vertex, face) of an axis-aligned L of two boxes: its principal axes are diagonal."""
  boxes = [((0.0, 0.0, 0.0), (1.0, 0.1, 0.1)), ((0.0, 0.1, 0.0), (0.1, 0.5, 0.1))]
  quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
  verts, faces = [], []
  for lo, hi in boxes:
    base = len(verts)
    verts += [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    for a, b, c, d in quads:
      faces += [(base + a, base + b, base + c), (base + a, base + c, base + d)]
  return " ".join(f"{x} {y} {z}" for x, y, z in verts), " ".join(f"{a} {b} {c}" for a, b, c in faces)


_VERTEX, _FACE = _l_shape()
XML = f"""
<mujoco>
  <asset>
    <mesh name="plank" vertex="{_VERTEX}" face="{_FACE}"/>
    <mesh name="tet" vertex="0 0 0  0.3 0 0  0 0.3 0  0 0 0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -2 1.2" xyaxes="1 0 0 0 0.5 1"/>
    <geom type="plane" size="2 2 0.1"/>
    <geom type="mesh" mesh="plank" pos="-0.4 0 0.2" rgba="0.8 0.5 0.2 1"/>
    <geom type="mesh" mesh="tet" pos="0.5 0.4 0" rgba="0.2 0.5 0.8 1"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("frame", ["auto", "mujoco"])
def test_mesh_bvh_constructor_replaces_bvhs(scene, frame):
  reference = scene(XML, cam_res=(32, 24), mesh_bvh_frame=frame).render()
  s = scene(XML, cam_res=(32, 24), mesh_bvh_constructor="sah", mesh_bvh_frame=frame).render()
  base = s.rc.base
  assert len(base.mesh_registry) == s.mjm.nmesh
  assert set(int(i) for i in base.mesh_bvh_id.numpy()) == set(base.mesh_registry)
  if frame == "auto":
    assert s.rc.mesh_bvh_frame is not None
  np.testing.assert_allclose(s.rgb(), reference.rgb(), atol=1.5 / 255)
