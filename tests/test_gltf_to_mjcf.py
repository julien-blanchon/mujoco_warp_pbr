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

"""Tests for tools/gltf_to_mjcf.py on a synthetic glTF quad."""

import io
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

pygltflib = pytest.importorskip("pygltflib", reason="needs the bench extra")
from gltf_to_mjcf import convert  # noqa: E402 - after the importorskip


def _png(rgb) -> bytes:
  buf = io.BytesIO()
  Image.fromarray(np.full((2, 2, 3), rgb, dtype=np.uint8)).save(buf, format="PNG")
  return buf.getvalue()


def _quad_glb(path: Path, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)):
  """A unit quad in glTF's XY plane, textured with base colour, normal, MR and occlusion maps."""
  pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
  normal = np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))
  uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
  idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
  images = [_png((200, 10, 10)), _png((128, 128, 255)), _png((0, 100, 200)), _png((50, 50, 50))]

  blob = b""
  views = []
  for data in [pos.tobytes(), normal.tobytes(), uv.tobytes(), idx.tobytes(), *images]:
    views.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
    blob += data + b"\0" * (-len(data) % 4)

  def acc(view, ctype, count, kind, **kw):
    return pygltflib.Accessor(bufferView=view, componentType=ctype, count=count, type=kind, **kw)

  doc = pygltflib.GLTF2(
    scene=0,
    scenes=[pygltflib.Scene(nodes=[0])],
    nodes=[pygltflib.Node(mesh=0, translation=[0.0, 2.0, 0.0], scale=list(scale))],
    meshes=[
      pygltflib.Mesh(
        primitives=[
          pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2), indices=3, material=0)
        ]
      )
    ],
    accessors=[
      acc(0, 5126, 4, "VEC3", min=[0, 0, 0], max=[1, 1, 0]),
      acc(1, 5126, 4, "VEC3"),
      acc(2, 5126, 4, "VEC2"),
      acc(3, 5123, 6, "SCALAR"),
    ],
    materials=[
      pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
          baseColorFactor=[1.0, 1.0, 1.0, 1.0],
          metallicFactor=0.25,
          roughnessFactor=0.75,
          baseColorTexture=pygltflib.TextureInfo(index=0),
          metallicRoughnessTexture=pygltflib.TextureInfo(index=2),
        ),
        normalTexture=pygltflib.NormalMaterialTexture(index=1),
        occlusionTexture=pygltflib.OcclusionTextureInfo(index=3),
      )
    ],
    textures=[pygltflib.Texture(source=i) for i in range(4)],
    images=[pygltflib.Image(bufferView=4 + i, mimeType="image/png") for i in range(4)],
    bufferViews=views,
    buffers=[pygltflib.Buffer(byteLength=len(blob))],
  )
  doc.set_binary_blob(blob)
  doc.save_binary(str(path))


def _world_vertices(m: mujoco.MjModel) -> np.ndarray:
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  adr, num = m.mesh_vertadr[0], m.mesh_vertnum[0]
  return m.mesh_vert[adr : adr + num] @ d.geom_xmat[0].reshape(3, 3).T + d.geom_xpos[0]


def test_material_layers_and_factors(tmp_path):
  _quad_glb(tmp_path / "quad.glb")
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))

  assert m.nmat == 1
  np.testing.assert_allclose(m.mat_metallic, [0.25])
  np.testing.assert_allclose(m.mat_roughness, [0.75])
  roles = mujoco.mjtTextureRole
  texid = m.mat_texid[0]
  for role in (roles.mjTEXROLE_RGB, roles.mjTEXROLE_NORMAL, roles.mjTEXROLE_ORM):
    assert texid[role] >= 0
  assert texid[roles.mjTEXROLE_EMISSIVE] == -1

  # ORM packs occlusion (R) with glTF metallicRoughness green/blue.
  orm = texid[roles.mjTEXROLE_ORM]
  adr, w, h, c = m.tex_adr[orm], m.tex_width[orm], m.tex_height[orm], m.tex_nchannel[orm]
  texel = m.tex_data[adr : adr + w * h * c].reshape(h, w, c)[0, 0]
  np.testing.assert_array_equal(texel[:3], [50, 100, 200])


def test_y_up_to_z_up_and_texcoords(tmp_path):
  _quad_glb(tmp_path / "quad.glb")
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))

  # glTF (x, y, z) -> MuJoCo (x, -z, y); node translation y=2 becomes z=2.
  verts = _world_vertices(m)
  np.testing.assert_allclose(verts.min(axis=0), [0, 0, 2], atol=1e-6)
  np.testing.assert_allclose(verts.max(axis=0), [1, 0, 3], atol=1e-6)

  # MuJoCo stores texcoords with glTF's top-left origin.
  uv = sorted(map(tuple, m.mesh_texcoord.round(6)))
  assert uv == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


@pytest.mark.parametrize("normalize", [None, 0.5])
def test_negative_scale_keeps_outward_winding(tmp_path, normalize):
  _quad_glb(tmp_path / "quad.glb", scale=(-1.0, 1.0, 1.0))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out", normalize=normalize)))

  # Mirroring flips handedness; the converter flips the winding so the face normal follows the
  # (inverse-transpose transformed) vertex normal, which stays glTF +z -> MuJoCo -y.
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  face = m.mesh_face[m.mesh_faceadr[0]]
  verts = m.mesh_vert[m.mesh_vertadr[0] :][:4] @ d.geom_xmat[0].reshape(3, 3).T
  n = np.cross(verts[face[1]] - verts[face[0]], verts[face[2]] - verts[face[0]])
  assert n[1] < 0.0

  if normalize is not None:
    extent = np.ptp(_world_vertices(m), axis=0)
    assert extent.max() == pytest.approx(normalize)
