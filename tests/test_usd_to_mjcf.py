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

"""Tests for tools/usd_to_mjcf.py on small synthetic USD stages."""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pxr")
from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdLux  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
from pxr import UsdShade  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import usd_to_mjcf  # noqa: E402

# Texel colours of the 2x2 test image, as the image is stored (row 0 at the top).
TOP_LEFT, TOP_RIGHT = (255, 0, 0), (0, 255, 0)
BOTTOM_LEFT, BOTTOM_RIGHT = (0, 0, 255), (255, 255, 255)


def _stage(path: Path, up: str = "Z", mpu: float = 1.0) -> Usd.Stage:
  stage = Usd.Stage.CreateNew(str(path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
  UsdGeom.SetStageMetersPerUnit(stage, mpu)
  stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
  return stage


def _quad(stage: Usd.Stage, path: str, nquad: int = 1) -> UsdGeom.Mesh:
  """`nquad` unit quads side by side along x in the z = 0 plane, st = xy within each quad."""
  mesh = UsdGeom.Mesh.Define(stage, path)
  points, counts, indices, st = [], [], [], []
  for q in range(nquad):
    base = len(points)
    points += [(q, 0, 0), (q + 1, 0, 0), (q + 1, 1, 0), (q, 1, 0)]
    counts.append(4)
    indices += [base, base + 1, base + 2, base + 3]
    st += [(0, 0), (1, 0), (1, 1), (0, 1)]
  mesh.CreatePointsAttr(points)
  mesh.CreateFaceVertexCountsAttr(counts)
  mesh.CreateFaceVertexIndicesAttr(indices)
  mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
  pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
  pv.Set(st)
  return mesh


def _preview_material(stage: Usd.Stage, path: str, colour=(0.5, 0.5, 0.5), texture: Path | None = None, uv_scale=None):
  material = UsdShade.Material.Define(stage, path)
  shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
  shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.2)
  if texture is None:
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
  else:
    reader = UsdShade.Shader.Define(stage, f"{path}/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_source = reader.ConnectableAPI()
    if uv_scale is not None:
      xform = UsdShade.Shader.Define(stage, f"{path}/Transform")
      xform.CreateIdAttr("UsdTransform2d")
      xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_source, "result")
      xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
      st_source = xform.ConnectableAPI()
    tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture.name))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_source, "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  return material


def _bind(prim, material):
  UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _world_corners(mjm: mujoco.MjModel, geom: int):
  """(world position, MuJoCo texcoord) of every face corner of a mesh geom."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  mesh = mjm.geom_dataid[geom]
  vert = mjm.mesh_vert[mjm.mesh_vertadr[mesh] : mjm.mesh_vertadr[mesh] + mjm.mesh_vertnum[mesh]]
  tc = mjm.mesh_texcoord[mjm.mesh_texcoordadr[mesh] : mjm.mesh_texcoordadr[mesh] + mjm.mesh_texcoordnum[mesh]]
  face = mjm.mesh_face[mjm.mesh_faceadr[mesh] : mjm.mesh_faceadr[mesh] + mjm.mesh_facenum[mesh]]
  facetc = mjm.mesh_facetexcoord[mjm.mesh_faceadr[mesh] : mjm.mesh_faceadr[mesh] + mjm.mesh_facenum[mesh]]
  world = vert @ mjd.geom_xmat[geom].reshape(3, 3).T + mjd.geom_xpos[geom]
  return world[face.reshape(-1)], tc[facetc.reshape(-1)] if len(tc) else None


def _texel(mjm: mujoco.MjModel, tex: int, uv) -> tuple:
  h, w, c = mjm.tex_height[tex], mjm.tex_width[tex], mjm.tex_nchannel[tex]
  data = mjm.tex_data[mjm.tex_adr[tex] : mjm.tex_adr[tex] + h * w * c].reshape(h, w, c)
  row = min(int(uv[1] * h), h - 1)
  col = min(int(uv[0] * w), w - 1)
  return tuple(int(x) for x in data[row, col, :3])


def test_textured_quad_uv_orientation(tmp_path):
  image = np.array([[TOP_LEFT, TOP_RIGHT], [BOTTOM_LEFT, BOTTOM_RIGHT]], dtype=np.uint8)
  Image.fromarray(image).save(tmp_path / "checker.png")
  stage = _stage(tmp_path / "quad.usda")
  mesh = _quad(stage, "/World/Quad")
  _bind(mesh.GetPrim(), _preview_material(stage, "/World/Looks/Tex", texture=tmp_path / "checker.png"))
  stage.Save()

  out = usd_to_mjcf.convert(tmp_path / "quad.usda", tmp_path / "out")
  mjm = mujoco.MjModel.from_xml_path(str(out))
  assert mjm.ngeom == 1 and mjm.nmat == 1
  mat = mjm.geom_matid[0]
  tex = mjm.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB]
  assert tex >= 0
  assert mjm.tex_colorspace[tex] == mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
  np.testing.assert_allclose(mjm.mat_metallic[mat], 0.2, atol=1e-6)
  np.testing.assert_allclose(mjm.mat_roughness[mat], 0.3, atol=1e-6)

  # USD st (0, 0) is the bottom-left of the image, (0, 1) the top-left.
  expected = {(0, 0): BOTTOM_LEFT, (1, 0): BOTTOM_RIGHT, (1, 1): TOP_RIGHT, (0, 1): TOP_LEFT}
  pos, uv = _world_corners(mjm, 0)
  inset = np.array([[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]])
  for (x, y), colour in expected.items():
    i = int(np.argmin(np.linalg.norm(pos - [x, y, 0], axis=1)))
    np.testing.assert_allclose(pos[i], [x, y, 0], atol=1e-5)
    # Sample a quarter texel inwards from the corner (texcoords run from 0 to 1 across the quad).
    step = inset[[(0, 0), (1, 0), (1, 1), (0, 1)].index((x, y))]
    assert _texel(mjm, tex, uv[i] + step * [1, -1]) == colour


def test_uv_transform_baked(tmp_path):
  Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "black.png")
  stage = _stage(tmp_path / "quad.usda")
  mesh = _quad(stage, "/World/Quad")
  _bind(mesh.GetPrim(), _preview_material(stage, "/World/Looks/Tex", texture=tmp_path / "black.png", uv_scale=(3, 2)))
  stage.Save()
  out = usd_to_mjcf.convert(tmp_path / "quad.usda", tmp_path / "out")
  obj = (out.parent / "meshes" / "Quad.obj").read_text().splitlines()
  vt = np.array([[float(v) for v in line.split()[1:]] for line in obj if line.startswith("vt ")])
  np.testing.assert_allclose(vt.max(axis=0), [3, 2])
  np.testing.assert_allclose(vt.min(axis=0), [0, 0])


def test_geom_subset_split(tmp_path):
  stage = _stage(tmp_path / "subset.usda")
  mesh = _quad(stage, "/World/Two", nquad=2)
  red = _preview_material(stage, "/World/Looks/Red", colour=(1, 0, 0))
  green = _preview_material(stage, "/World/Looks/Green", colour=(0, 1, 0))
  _bind(mesh.GetPrim(), red)
  binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
  subset = binding.CreateMaterialBindSubset("second", [1], UsdGeom.Tokens.face)
  _bind(subset.GetPrim(), green)
  stage.Save()

  mjm = mujoco.MjModel.from_xml_path(str(usd_to_mjcf.convert(tmp_path / "subset.usda", tmp_path / "out")))
  assert mjm.ngeom == 2
  colours = {}
  for g in range(mjm.ngeom):
    pos, _ = _world_corners(mjm, g)
    assert mjm.mesh_facenum[mjm.geom_dataid[g]] == 2  # one quad each
    colours[int(pos[:, 0].min().round())] = tuple(mjm.mat_rgba[mjm.geom_matid[g], :3])
  assert colours == {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}  # face 1 (x in [1, 2]) is the subset


def test_lights(tmp_path):
  stage = _stage(tmp_path / "lights.usda")
  _quad(stage, "/World/Floor")
  sphere = UsdLux.SphereLight.Define(stage, "/World/Bulb")
  sphere.CreateRadiusAttr(0.1)
  sphere.CreateIntensityAttr(1000.0)
  sphere.CreateExposureAttr(1.0)
  sphere.CreateColorAttr(Gf.Vec3f(1.0, 0.5, 0.25))
  sphere.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.5, 2.0))
  sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
  sun.CreateIntensityAttr(500.0)
  sun.AddRotateXOp().Set(30.0)  # -Z tilted towards +y
  rect = UsdLux.RectLight.Define(stage, "/World/Panel")
  rect.CreateWidthAttr(2.0)
  rect.CreateHeightAttr(0.5)
  rect.CreateIntensityAttr(100.0)
  rect.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.0))
  stage.Save()

  out = usd_to_mjcf.convert(tmp_path / "lights.usda", tmp_path / "out")
  mjm = mujoco.MjModel.from_xml_path(str(out))
  # No camera: Omniverse's default exposure (f/5, 1/50 s, ISO 100) at the renderer's EV100.
  scale = 0.02 / 25.0 * 1.2 * 2.0**usd_to_mjcf.DEFAULT_EV100
  lights = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_LIGHT, i): i for i in range(mjm.nlight)}

  bulb = lights["Bulb"]
  assert mjm.light_type[bulb] == mujoco.mjtLightType.mjLIGHT_POINT
  np.testing.assert_allclose(mjm.light_pos[bulb], [0.5, 0.5, 2.0], atol=1e-6)
  np.testing.assert_allclose(mjm.light_bulbradius[bulb], 0.1, rtol=1e-6)
  np.testing.assert_allclose(mjm.light_diffuse[bulb], [1.0, 0.5, 0.25], rtol=1e-6)
  # Candela: radiance (intensity x 2^exposure) times the sphere's projected area.
  np.testing.assert_allclose(mjm.light_intensity[bulb], 2000.0 * math.pi * 0.01 * scale, rtol=1e-5)

  s = lights["Sun"]
  assert mjm.light_type[s] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  np.testing.assert_allclose(mjm.light_dir[s], [0.0, math.sin(math.radians(30)), -math.cos(math.radians(30))], atol=1e-6)
  np.testing.assert_allclose(mjm.light_intensity[s], 500.0 * scale, rtol=1e-5)

  panel = lights["Panel"]
  assert mjm.light_type[panel] == mujoco.mjtLightType.mjLIGHT_SPOT
  np.testing.assert_allclose(mjm.light_dir[panel], [0, 0, -1], atol=1e-6)
  np.testing.assert_allclose(mjm.light_intensity[panel], 100.0 * 1.0 * scale, rtol=1e-5)
  np.testing.assert_allclose(mjm.light_bulbradius[panel], math.sqrt(1.0 / math.pi), rtol=1e-5)
  # The scene's own lights replace the headlight.
  np.testing.assert_allclose(mjm.vis.headlight.diffuse, 0.0)


def test_instance_proxies(tmp_path):
  stage = _stage(tmp_path / "instances.usda")
  proto = stage.CreateClassPrim("/_proto")
  _quad(stage, "/_proto/Quad")
  _bind(stage.GetPrimAtPath("/_proto/Quad"), _preview_material(stage, "/_proto/Looks/Grey"))
  for i, offset in enumerate([(0, 0, 0), (5, 0, 1)]):
    inst = UsdGeom.Xform.Define(stage, f"/World/Inst{i}")
    inst.AddTranslateOp().Set(Gf.Vec3d(*offset))
    inst.GetPrim().GetReferences().AddInternalReference(proto.GetPath())
    inst.GetPrim().SetInstanceable(True)
  stage.Save()
  assert Usd.Stage.Open(str(tmp_path / "instances.usda")).GetPrimAtPath("/World/Inst1/Quad").IsInstanceProxy()

  mjm = mujoco.MjModel.from_xml_path(str(usd_to_mjcf.convert(tmp_path / "instances.usda", tmp_path / "out")))
  assert mjm.ngeom == 2
  lows = sorted(tuple(_world_corners(mjm, g)[0].min(axis=0).round(5)) for g in range(mjm.ngeom))
  assert lows == [(0.0, 0.0, 0.0), (5.0, 0.0, 1.0)]
  assert mjm.nmat == 1  # identical prototype materials are shared


def test_y_up_units_and_primitives(tmp_path):
  stage = _stage(tmp_path / "prims.usda", up="Y", mpu=0.01)
  cube = UsdGeom.Cube.Define(stage, "/World/Box")
  cube.CreateSizeAttr(100.0)  # 1 m
  cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 50.0, 0.0))  # 0.5 m up
  cube.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 1.0))
  sphere = UsdGeom.Sphere.Define(stage, "/World/Ball")
  sphere.CreateRadiusAttr(10.0)
  sphere.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -200.0))  # USD -z = MuJoCo +y
  capsule = UsdGeom.Capsule.Define(stage, "/World/Pill")
  capsule.CreateRadiusAttr(5.0)
  capsule.CreateHeightAttr(40.0)
  capsule.CreateAxisAttr(UsdGeom.Tokens.x)
  stage.Save()

  mjm = mujoco.MjModel.from_xml_path(str(usd_to_mjcf.convert(tmp_path / "prims.usda", tmp_path / "out")))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  box = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "Box")
  assert mjm.geom_type[box] == mujoco.mjtGeom.mjGEOM_BOX
  np.testing.assert_allclose(mjd.geom_xpos[box], [0, 0, 0.5], atol=1e-6)
  # USD y (scaled by 2) is MuJoCo z.
  half = np.abs(mjd.geom_xmat[box].reshape(3, 3)) @ mjm.geom_size[box]
  np.testing.assert_allclose(half, [0.5, 0.5, 1.0], atol=1e-6)
  ball = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "Ball")
  assert mjm.geom_type[ball] == mujoco.mjtGeom.mjGEOM_SPHERE
  np.testing.assert_allclose(mjm.geom_size[ball, 0], 0.1, atol=1e-6)
  np.testing.assert_allclose(mjd.geom_xpos[ball], [0, 2.0, 0], atol=1e-6)
  pill = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "Pill")
  assert mjm.geom_type[pill] == mujoco.mjtGeom.mjGEOM_CAPSULE
  np.testing.assert_allclose(mjm.geom_size[pill, :2], [0.05, 0.2], atol=1e-6)
  np.testing.assert_allclose(np.abs(mjd.geom_xmat[pill].reshape(3, 3)[:, 2]), [1, 0, 0], atol=1e-6)


def test_joint_posing(tmp_path):
  stage = _stage(tmp_path / "arm.usda")
  base = UsdGeom.Xform.Define(stage, "/World/Base")
  UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
  link = UsdGeom.Xform.Define(stage, "/World/Base/Link")
  UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
  link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
  cube = UsdGeom.Cube.Define(stage, "/World/Base/Link/Tip")
  cube.CreateSizeAttr(0.1)
  cube.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
  joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/hinge")
  joint.CreateBody0Rel().SetTargets([base.GetPath()])
  joint.CreateBody1Rel().SetTargets([link.GetPath()])
  joint.CreateAxisAttr("Z")
  joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 1.0))
  joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
  stage.Save()

  out = usd_to_mjcf.convert(tmp_path / "arm.usda", tmp_path / "out", joint_positions={"hinge": math.pi / 2})
  mjm = mujoco.MjModel.from_xml_path(str(out))
  tip = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "Tip")
  np.testing.assert_allclose(mjm.geom_pos[tip], [0.0, 1.0, 1.0], atol=1e-6)


def test_omnipbr_wrapper_mdl(tmp_path):
  (tmp_path / "Wood").mkdir()
  Image.fromarray(np.full((4, 4, 3), 90, dtype=np.uint8)).save(tmp_path / "Wood" / "Wood_BaseColor.png")
  Image.fromarray(np.full((4, 4, 3), (255, 140, 0), dtype=np.uint8)).save(tmp_path / "Wood" / "Wood_ORM.png")
  (tmp_path / "Wood.mdl").write_text(
    """mdl 1.4;
import ::OmniPBR::OmniPBR;
export material Wood(*)
[[ ::anno::display_name("Omni PBR ") ]]
 = ::OmniPBR::OmniPBR(
    diffuse_texture: texture_2d("./Wood/Wood_BaseColor.png" /* tag 1 */, ::tex::gamma_srgb),
    reflection_roughness_constant: 0.5f,
    enable_ORM_texture: true,
    ORM_texture: texture_2d("./Wood/Wood_ORM.png", ::tex::gamma_linear),
    texture_scale: float2(1.f));
"""
  )
  stage = _stage(tmp_path / "mdl.usda")
  mesh = _quad(stage, "/World/Quad")
  material = UsdShade.Material.Define(stage, "/World/Looks/Wood")
  shader = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Shader")
  shader.SetSourceAsset(Sdf.AssetPath("./Wood.mdl"), "mdl")
  shader.SetSourceAssetSubIdentifier("Wood", "mdl")
  shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(4, 4))
  material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
  _bind(mesh.GetPrim(), material)
  stage.Save()

  mjm = mujoco.MjModel.from_xml_path(str(usd_to_mjcf.convert(tmp_path / "mdl.usda", tmp_path / "out")))
  mat = mjm.geom_matid[0]
  assert mjm.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0
  orm = mjm.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_ORM]
  assert orm >= 0 and mjm.tex_colorspace[orm] == mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
  np.testing.assert_allclose([mjm.mat_roughness[mat], mjm.mat_metallic[mat]], [1.0, 1.0])
  _, uv = _world_corners(mjm, 0)
  np.testing.assert_allclose(uv.max(axis=0) - uv.min(axis=0), [4, 4], atol=1e-6)  # USD input overrides the MDL
