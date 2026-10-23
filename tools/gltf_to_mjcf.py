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

"""Convert a glTF 2.0 asset (.gltf / .glb) into a visual-only MJCF model.

The output uses native MuJoCo material fields only:

  baseColorFactor              -> material rgba
  metallicFactor/roughness     -> material metallic / roughness
  baseColorTexture             -> texture layer role="rgb" (role="rgba" for BLEND/MASK), sRGB
  normalTexture                -> texture layer role="normal", linear
  occlusion + metallicRoughness-> one packed texture layer role="orm" (R=AO, G=rough, B=metal)
  emissiveTexture/Factor       -> texture layer role="emissive", sRGB, factor baked in

Every glTF triangle primitive becomes one OBJ mesh (node transforms baked in, Y-up converted
to MuJoCo's Z-up) and one static, non-colliding mesh geom.

Not converted: KHR_* extensions, texture transforms, TEXCOORD_1, vertex colours, double
sidedness, alpha cutoff (MASK is written as blended alpha), skins, morph targets, cameras.

With --normalize SIZE the model is centred in x/y, rests on z = 0, and is scaled so that its
largest bounding-box extent equals SIZE metres (applied after --scale).

Usage:
  python tools/gltf_to_mjcf.py MODEL.glb OUT_DIR [--name NAME] [--scale S] [--normalize SIZE]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pygltflib
from PIL import Image

if TYPE_CHECKING:
  from shape_extensions import IntVar

type Matrix4 = np.ndarray[[4, 4], np.dtype[np.float64]]

_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_TRIANGLES = 4

# glTF is Y-up; MuJoCo is Z-up: (x, y, z) -> (x, -z, y).
_Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


class _Gltf:
  """Accessor, image and transform helpers around a loaded pygltflib document."""

  def __init__(self, path: Path):
    self.path = path
    self.doc = pygltflib.GLTF2().load(str(path))
    self._buffers = {}

  def _buffer(self, index: int) -> bytes:
    if index not in self._buffers:
      buf = self.doc.buffers[index]
      if buf.uri is None:
        self._buffers[index] = self.doc.binary_blob()
      elif buf.uri.startswith("data:"):
        self._buffers[index] = self.doc.decode_data_uri(buf.uri)
      else:
        self._buffers[index] = (self.path.parent / buf.uri).read_bytes()
    return self._buffers[index]

  def _view_bytes(self, view_index: int) -> bytes:
    view = self.doc.bufferViews[view_index]
    data = self._buffer(view.buffer)
    return data[view.byteOffset or 0 : (view.byteOffset or 0) + view.byteLength]

  def accessor(self, index: int) -> np.ndarray:
    acc = self.doc.accessors[index]
    if acc.sparse is not None:
      raise NotImplementedError("sparse accessors are not supported")
    dtype = np.dtype(_COMPONENT_DTYPE[acc.componentType])
    ncomp = _TYPE_SIZE[acc.type]
    view = self.doc.bufferViews[acc.bufferView]
    data = self._view_bytes(acc.bufferView)
    # numpy stubs: np.dtype has no itemsize; glTF only normalises integer components (np.iinfo).
    stride = view.byteStride or dtype.itemsize * ncomp  # pyrefly: ignore[missing-attribute]
    offset = acc.byteOffset or 0
    rows = np.lib.stride_tricks.as_strided(
      np.frombuffer(data, dtype=np.uint8, offset=offset),
      shape=(acc.count, dtype.itemsize * ncomp),  # pyrefly: ignore[missing-attribute]
      strides=(stride, 1),
    )
    out = np.ascontiguousarray(rows).view(dtype).reshape(acc.count, ncomp)
    if acc.normalized:
      out = out.astype(np.float32) / np.iinfo(dtype).max  # pyrefly: ignore[no-matching-overload]
    return out if ncomp > 1 else out[:, 0]

  def image(self, texture_index: int | None) -> Image.Image:
    if texture_index is None:
      raise ValueError(f"{self.path}: texture reference without an index")
    image = self.doc.images[self.doc.textures[texture_index].source]
    if image.bufferView is not None:
      return Image.open(io.BytesIO(self._view_bytes(image.bufferView)))
    if image.uri.startswith("data:"):
      return Image.open(io.BytesIO(self.doc.decode_data_uri(image.uri)))
    return Image.open(self.path.parent / image.uri)

  def node_matrix(self, node: pygltflib.Node) -> Matrix4:
    if node.matrix is not None:
      return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # glTF is column-major.
    t = np.eye(4)
    t[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    x, y, z, w = node.rotation or [0.0, 0.0, 0.0, 1.0]
    r = np.eye(4)
    r[:3, :3] = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    s = np.diag([*(node.scale or [1.0, 1.0, 1.0]), 1.0])  # pyrefly: ignore[no-matching-overload] - numpy stubs: diag(list)
    return t @ r @ s


def _write_obj[N: IntVar](
  path: Path,
  pos: np.ndarray[[N, 3], np.dtype[Any]],
  normal: np.ndarray[[N, 3], np.dtype[Any]] | None,
  uv: np.ndarray[[N, 2], np.dtype[Any]] | None,
  faces: np.ndarray[[int, 3], np.dtype[np.int64]],
):
  lines = [f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}" for p in pos]
  if uv is not None:
    # MuJoCo flips v when loading OBJ, so write OBJ's bottom-left convention.
    lines += [f"vt {t[0]:.7g} {1.0 - t[1]:.7g}" for t in uv]
  if normal is not None:
    lines += [f"vn {n[0]:.7g} {n[1]:.7g} {n[2]:.7g}" for n in normal]
  for f in faces + 1:
    if uv is not None and normal is not None:
      lines.append("f " + " ".join(f"{i}/{i}/{i}" for i in f))
    elif uv is not None:
      lines.append("f " + " ".join(f"{i}/{i}" for i in f))
    elif normal is not None:
      lines.append("f " + " ".join(f"{i}//{i}" for i in f))
    else:
      lines.append("f " + " ".join(str(i) for i in f))
  path.write_text("\n".join(lines) + "\n")


class _Converter:
  def __init__(self, gltf: _Gltf, out_dir: Path, name: str, scale: float, normalize: float | None):
    self.gltf = gltf
    self.out_dir = out_dir
    self.name = name
    self.scale = scale
    self.normalize = normalize
    self.primitives = []  # (name, pos, normal, uv, faces) before normalisation
    self.textures = []  # (name, file, colorspace)
    self.materials = {}  # gltf material index -> MJCF <material> element string
    self.meshes = []  # (name, file)
    self.geoms = []  # MJCF <geom> element strings
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)

  def _save_texture(self, name: str, image: Image.Image, colorspace: str) -> str:
    file = f"textures/{name}.png"
    image.save(self.out_dir / file)
    self.textures.append((name, file, colorspace))
    return name

  def _orm_texture(self, index: int, mat: pygltflib.Material) -> str | None:
    pbr = mat.pbrMetallicRoughness
    mr = pbr.metallicRoughnessTexture if pbr else None
    occ = mat.occlusionTexture
    if mr is None and occ is None:
      return None
    mr_img = np.asarray(self.gltf.image(mr.index).convert("RGB")) if mr else None
    occ_img = np.asarray(self.gltf.image(occ.index).convert("RGB")) if occ else None
    shape = next(img for img in (mr_img, occ_img) if img is not None).shape
    orm = np.full(shape, 255, dtype=np.uint8)
    if mr_img is not None:
      orm[..., 1:] = mr_img[..., 1:]
    if occ is not None and occ_img is not None:
      if occ_img.shape != shape:
        occ_img = np.asarray(Image.fromarray(occ_img).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
      strength = occ.strength if occ.strength is not None else 1.0
      orm[..., 0] = np.clip(255.0 - strength * (255.0 - occ_img[..., 0]), 0, 255).astype(np.uint8)
    return self._save_texture(f"mat{index}_orm", Image.fromarray(orm), "linear")

  def _emissive_texture(self, index: int, mat: pygltflib.Material) -> str | None:
    factor = np.array(mat.emissiveFactor or [0.0, 0.0, 0.0], dtype=np.float32)
    if not factor.any():
      return None
    if mat.emissiveTexture is not None:
      img = np.asarray(self.gltf.image(mat.emissiveTexture.index).convert("RGB"), dtype=np.float32) / 255.0
    else:
      img = np.ones((1, 1, 3), dtype=np.float32)
    # Bake the factor in sRGB space via linear: decode, scale, re-encode.
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4) * factor
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    out = Image.fromarray(np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return self._save_texture(f"mat{index}_emissive", out, "sRGB")

  def material(self, index: int | None) -> str | None:
    if index is None:
      return None
    if index in self.materials:
      return f"mat{index}"
    mat = self.gltf.doc.materials[index]
    pbr = mat.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
    rgba = list(pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0])
    alpha_mode = mat.alphaMode or "OPAQUE"
    if alpha_mode == "OPAQUE":
      rgba[3] = 1.0

    layers = []
    if pbr.baseColorTexture is not None:
      role = "rgb" if alpha_mode == "OPAQUE" else "rgba"
      mode = "RGB" if role == "rgb" else "RGBA"
      layers.append(
        (self._save_texture(f"mat{index}_basecolor", self.gltf.image(pbr.baseColorTexture.index).convert(mode), "sRGB"), role)
      )
    if mat.normalTexture is not None:
      layers.append(
        (self._save_texture(f"mat{index}_normal", self.gltf.image(mat.normalTexture.index).convert("RGB"), "linear"), "normal")
      )
    orm = self._orm_texture(index, mat)
    if orm is not None:
      layers.append((orm, "orm"))
    emissive = self._emissive_texture(index, mat)
    if emissive is not None:
      layers.append((emissive, "emissive"))

    metallic = 1.0 if pbr.metallicFactor is None else pbr.metallicFactor
    roughness = 1.0 if pbr.roughnessFactor is None else pbr.roughnessFactor
    attrs = (
      f'name="mat{index}" rgba="{" ".join(f"{c:.6g}" for c in rgba)}" metallic="{metallic:.6g}" roughness="{roughness:.6g}"'
    )
    if emissive is not None:
      attrs += ' emission="1"'
    body = "".join(f'\n      <layer texture="{tex}" role="{role}"/>' for tex, role in layers)
    self.materials[index] = f"    <material {attrs}>{body}\n    </material>" if body else f"    <material {attrs}/>"
    return f"mat{index}"

  def primitive(self, mesh_index: int, prim_index: int, prim: pygltflib.Primitive, world: Matrix4):
    if (prim.mode if prim.mode is not None else _TRIANGLES) != _TRIANGLES:
      print(f"  skipping mesh {mesh_index} primitive {prim_index}: mode {prim.mode} is not TRIANGLES")
      return
    attr = prim.attributes
    if attr.POSITION is None:
      raise ValueError(f"mesh {mesh_index} primitive {prim_index} has no POSITION attribute")
    pos = self.gltf.accessor(attr.POSITION).astype(np.float64)
    normal = self.gltf.accessor(attr.NORMAL).astype(np.float64) if attr.NORMAL is not None else None
    uv = self.gltf.accessor(attr.TEXCOORD_0).astype(np.float64) if attr.TEXCOORD_0 is not None else None
    if prim.indices is not None:
      faces = self.gltf.accessor(prim.indices).astype(np.int64).reshape(-1, 3)
    else:
      faces = np.arange(len(pos)).reshape(-1, 3)

    pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3] * self.scale
    if normal is not None:
      normal = normal @ np.linalg.inv(world[:3, :3])  # inverse transpose, row vectors.
      normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    if np.linalg.det(world[:3, :3]) < 0.0:
      faces = faces[:, ::-1]

    name = f"mesh{mesh_index}_{prim_index}_{len(self.primitives)}"
    self.primitives.append((name, pos, normal, uv, faces))
    mat = self.material(prim.material)
    mat_attr = f' material="{mat}"' if mat else ""
    self.geoms.append(f'    <geom type="mesh" mesh="{name}"{mat_attr} contype="0" conaffinity="0" group="1" mass="0"/>')

  def node(self, node_index: int, parent: Matrix4):
    node = self.gltf.doc.nodes[node_index]
    world = parent @ self.gltf.node_matrix(node)
    if node.mesh is not None:
      for prim_index, prim in enumerate(self.gltf.doc.meshes[node.mesh].primitives):
        self.primitive(node.mesh, prim_index, prim, world)
    for child in node.children or []:
      self.node(child, world)

  def run(self) -> Path:
    doc = self.gltf.doc
    scene = doc.scenes[doc.scene if doc.scene is not None else 0]
    for node_index in scene.nodes:
      self.node(node_index, _Y_UP_TO_Z_UP)

    offset, factor = np.zeros(3), 1.0
    if self.normalize is not None and self.primitives:
      lower = np.min([p[1].min(axis=0) for p in self.primitives], axis=0)
      upper = np.max([p[1].max(axis=0) for p in self.primitives], axis=0)
      factor = self.normalize / max(float((upper - lower).max()), 1e-12)
      offset = -np.array([0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), lower[2]])
    for name, pos, normal, uv, faces in self.primitives:
      file = f"meshes/{name}.obj"
      _write_obj(self.out_dir / file, (pos + offset) * factor, normal, uv, faces)
      self.meshes.append((name, file))

    textures = "\n".join(f'    <texture name="{n}" type="2d" file="{f}" colorspace="{c}"/>' for n, f, c in self.textures)
    meshes = "\n".join(f'    <mesh name="{n}" file="{f}" inertia="shell"/>' for n, f in self.meshes)
    xml = f"""<mujoco model="{self.name}">
  <!-- Converted from {self.gltf.path.name} by tools/gltf_to_mjcf.py -->
  <compiler meshdir="." texturedir="."/>
  <asset>
{textures}
{chr(10).join(self.materials.values())}
{meshes}
  </asset>
  <worldbody>
{chr(10).join(self.geoms)}
  </worldbody>
</mujoco>
"""
    out = self.out_dir / f"{self.name}.xml"
    out.write_text(xml)
    return out


def convert(path: Path, out_dir: Path, name: str | None = None, scale: float = 1.0, normalize: float | None = None) -> Path:
  """Convert a glTF/GLB file and return the path of the written MJCF file."""
  out_dir.mkdir(parents=True, exist_ok=True)
  return _Converter(_Gltf(path), out_dir, name or path.stem, scale, normalize).run()


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("gltf", type=Path)
  parser.add_argument("out_dir", type=Path)
  parser.add_argument("--name")
  parser.add_argument("--scale", type=float, default=1.0)
  parser.add_argument("--normalize", type=float, help="target largest extent in metres")
  args = parser.parse_args()
  print(convert(args.gltf, args.out_dir, args.name, args.scale, args.normalize))


if __name__ == "__main__":
  main()
