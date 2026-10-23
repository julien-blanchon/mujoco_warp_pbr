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

"""Builds a Mitsuba 3 scene equivalent to a MuJoCo scene rendered by mujoco_warp_pbr.

Covers spheres, ellipsoids, planes, boxes, capsules, cylinders and meshes; the renderer's BSDF
or the principled BSDF with base colour / roughness / metallic / normal textures; directional,
point and spot lights (point-like: bulb radii are ignored), a constant ambient and an equirect
environment map. Material values come from
the PbrRenderContext so both renderers see identical inputs. Requires a Mitsuba variant to be set.
"""

import mitsuba as mi
import mujoco
import numpy as np

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import lights as pbr_lights
from mujoco_warp_pbr._src.texture import is_srgb

ROLE = mujoco.mjtTextureRole
FLIP_NORMAL_GREEN = False


def _transform(pos: np.ndarray, mat: np.ndarray, scale) -> "mi.ScalarTransform4f":
  m = np.eye(4)
  m[:3, :3] = mat.reshape(3, 3) * np.asarray(scale)[None, :]
  m[:3, 3] = pos
  return mi.ScalarTransform4f(m.tolist())


def _frame_z(axis: np.ndarray) -> np.ndarray:
  """A rotation whose third column is the unit vector `axis`."""
  z = axis / np.linalg.norm(axis)
  x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
  x /= np.linalg.norm(x)
  return np.stack([x, np.cross(z, x), z], axis=1)


def _tube(pos: np.ndarray, mat: np.ndarray, radius: float, half_length: float, caps: str) -> list[dict]:
  """Shapes of a capsule (caps="sphere") or a capped cylinder (caps="disk") along the geom's z."""
  axis = mat.reshape(3, 3)[:, 2]
  p0, p1 = pos - axis * half_length, pos + axis * half_length
  shapes = [{"type": "cylinder", "p0": p0.tolist(), "p1": p1.tolist(), "radius": radius}]
  for end, outward in ((p0, -axis), (p1, axis)):
    if caps == "sphere":
      shapes.append({"type": "sphere", "center": end.tolist(), "radius": radius})
    else:
      shapes.append({"type": "disk", "to_world": _transform(end, _frame_z(outward), [radius, radius, 1.0])})
  return shapes


def _texture_array(mjm: mujoco.MjModel, rc: mjpbr.PbrRenderContext, tex_id: int) -> np.ndarray:
  """The texture as uploaded to the GPU (sRGB decoded for colour textures), float32 (h, w, 4)."""
  w, h, nc = mjm.tex_width[tex_id], mjm.tex_height[tex_id], mjm.tex_nchannel[tex_id]
  raw = mjm.tex_data[mjm.tex_adr[tex_id] : mjm.tex_adr[tex_id] + w * h * nc].reshape(h, w, nc).astype(np.float32) / 255
  rgba = np.ones((h, w, 4), np.float32)
  if nc == 1:
    rgba[..., :3] = raw
  else:
    rgba[..., :nc] = raw
  if is_srgb(mjm, tex_id):
    rgba[..., :3] = np.where(rgba[..., :3] <= 0.04045, rgba[..., :3] / 12.92, ((rgba[..., :3] + 0.055) / 1.055) ** 2.4)
  return rgba


def _bitmap(array: np.ndarray) -> dict:
  return {
    "type": "bitmap",
    "bitmap": mi.Bitmap(np.ascontiguousarray(array, dtype=np.float32)),
    "raw": True,
    "filter_type": "bilinear",
  }


def _bsdf(mjm, rc, geom: int, model: str) -> dict:
  metallic = float(rc.geom_metallic.numpy()[0, geom])
  roughness = float(rc.geom_roughness.numpy()[0, geom])
  reflectance = float(rc.geom_reflectance.numpy()[0, geom])
  mat = mjm.geom_matid[geom]
  rgba = mjm.geom_rgba[geom]
  if mat >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
    rgba = mjm.mat_rgba[mat]
  base_color = {"type": "rgb", "value": rgba[:3].tolist()}
  rough: dict | float = max(roughness, 0.045)
  metal: dict | float = metallic
  normal_map = None
  if mat >= 0 and rc.features.textures:
    tex = mjm.mat_texid[mat]
    color_id = tex[ROLE.mjTEXROLE_RGB] if tex[ROLE.mjTEXROLE_RGB] >= 0 else tex[ROLE.mjTEXROLE_RGBA]
    if color_id >= 0:
      base_color = _bitmap(_texture_array(mjm, rc, color_id)[..., :3] * rgba[:3])
    if tex[ROLE.mjTEXROLE_ORM] >= 0:
      orm = _texture_array(mjm, rc, tex[ROLE.mjTEXROLE_ORM])
      rough = _bitmap(np.maximum(orm[..., 1:2] * roughness, 0.045))
      metal = _bitmap(orm[..., 2:3] * metallic)
    if tex[ROLE.mjTEXROLE_NORMAL] >= 0:
      normal = _texture_array(mjm, rc, tex[ROLE.mjTEXROLE_NORMAL])[..., :3]
      if FLIP_NORMAL_GREEN:
        normal = normal * np.array([1.0, -1.0, 1.0]) + np.array([0.0, 1.0, 0.0])
      normal_map = _bitmap(normal)
  if model == "filament":
    principled = {
      "type": "filament",
      "base_color": base_color,
      "metallic": metal,
      "roughness": rough,
      "reflectance": reflectance,
    }
  else:
    principled = {
      "type": "principled",
      "base_color": base_color,
      "metallic": metal,
      "roughness": rough,
      # Mitsuba: F0 = 0.08 · specular; ours: F0 = 0.16 · reflectance².
      "specular": min(2.0 * reflectance * reflectance, 1.0),
    }
  if normal_map is not None:
    return {"type": "normalmap", "normalmap": normal_map, "bsdf": principled}
  return {"type": "twosided", "bsdf": principled}


def _emitter(mjm, rc, geom: int) -> dict | None:
  """Area emitter matching the renderer's display-relative emission (exposure 1)."""
  mat = mjm.geom_matid[geom]
  if mat < 0 or mjm.mat_emission[mat] <= 0:
    return None
  exposure = float(rc.exposure.numpy()[0])
  emission = float(mjm.mat_emission[mat]) / exposure
  tex = mjm.mat_texid[mat, ROLE.mjTEXROLE_EMISSIVE]
  if rc.features.textures and tex >= 0:
    return {"type": "area", "radiance": _bitmap(_texture_array(mjm, rc, tex)[..., :3] * emission)}
  rgba = mjm.geom_rgba[geom] if not np.allclose(mjm.geom_rgba[geom], [0.5, 0.5, 0.5, 1]) else mjm.mat_rgba[mat]
  return {"type": "area", "radiance": {"type": "rgb", "value": (rgba[:3] * emission).tolist()}}


def _mesh_shape(mjm, mjd, geom: int, name: str, bsdf: dict, emitter: dict | None) -> "mi.Mesh":
  mesh = mjm.geom_dataid[geom]
  face = mjm.mesh_face[mjm.mesh_faceadr[mesh] : mjm.mesh_faceadr[mesh] + mjm.mesh_facenum[mesh]]
  vert = mjm.mesh_vert[mjm.mesh_vertadr[mesh] : mjm.mesh_vertadr[mesh] + mjm.mesh_vertnum[mesh]]
  xmat = mjd.geom_xmat[geom].reshape(3, 3)
  pos = (vert[face.ravel()] @ xmat.T + mjd.geom_xpos[geom]).astype(np.float32)
  nface = face.shape[0]
  facenormal = mjm.mesh_facenormal[mjm.mesh_faceadr[mesh] : mjm.mesh_faceadr[mesh] + nface]
  normal = (mjm.mesh_normal[mjm.mesh_normaladr[mesh] + facenormal.ravel()] @ xmat.T).astype(np.float32)
  has_uv = mjm.mesh_texcoordadr[mesh] >= 0
  props = mi.Properties()
  props["bsdf"] = mi.load_dict(bsdf)
  if emitter is not None:
    props["emitter"] = mi.load_dict(emitter)
  out = mi.Mesh(name, 3 * nface, nface, props, has_vertex_normals=True, has_vertex_texcoords=bool(has_uv))
  params = mi.traverse(out)
  params["vertex_positions"] = mi.Float(pos.ravel())
  params["vertex_normals"] = mi.Float(normal.ravel())
  params["faces"] = mi.UInt(np.arange(3 * nface, dtype=np.uint32))
  if has_uv:
    facetc = mjm.mesh_facetexcoord[mjm.mesh_faceadr[mesh] : mjm.mesh_faceadr[mesh] + nface]
    uv = mjm.mesh_texcoord[mjm.mesh_texcoordadr[mesh] + facetc.ravel()].astype(np.float32)
    # MuJoCo and Mitsuba's bitmap lookup both use a top-left texture origin.
    params["vertex_texcoords"] = mi.Float(uv.ravel())
  params.update()
  return out


def scene_dict(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  rc: mjpbr.PbrRenderContext,
  camera: int,
  width: int,
  height: int,
  spp: int,
  envmap: np.ndarray | None = None,
  env_to_world: "mi.ScalarTransform4f | None" = None,
  integrator: str = "direct",
  max_depth: int = 2,
  bsdf: str = "filament",
) -> dict:
  """Mitsuba scene dict for the model state `mjd` seen from `camera`.

  bsdf: "filament" (the renderer's BRDF, see filament_bsdf.py) or "principled" (Mitsuba's
  Disney BSDF, an independent material model).
  """
  zaxis = mjd.cam_xmat[camera].reshape(3, 3)[:, 2]
  yaxis = mjd.cam_xmat[camera].reshape(3, 3)[:, 1]
  origin = mjd.cam_xpos[camera]
  scene = {
    "type": "scene",
    "integrator": {"type": integrator} if integrator == "direct" else {"type": integrator, "max_depth": max_depth},
    "sensor": {
      "type": "perspective",
      "fov_axis": "y",
      "fov": float(mjm.cam_fovy[camera]),
      "near_clip": 1e-3,
      "to_world": mi.ScalarTransform4f().look_at(origin=origin.tolist(), target=(origin - zaxis).tolist(), up=yaxis.tolist()),
      "film": {"type": "hdrfilm", "width": width, "height": height, "rfilter": {"type": "box"}, "pixel_format": "rgb"},
      "sampler": {"type": "independent", "sample_count": spp},
    },
  }

  for g in range(mjm.ngeom):
    if mjm.geom_rgba[g, 3] == 0.0:
      continue
    gtype = mjm.geom_type[g]
    size = mjm.geom_size[g]
    pos, mat = mjd.geom_xpos[g], mjd.geom_xmat[g]
    bsdf_dict = _bsdf(mjm, rc, g, bsdf)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
      shape = {"type": "sphere", "center": pos.tolist(), "radius": float(size[0])}
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
      shape = {"type": "sphere", "to_world": _transform(pos, mat, size)}
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
      half = [s if s > 0 else 100.0 for s in size[:2]]
      shape = {"type": "rectangle", "to_world": _transform(pos, mat, [half[0], half[1], 1.0])}
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
      shape = {"type": "cube", "to_world": _transform(pos, mat, size)}
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
      scene[f"geom{g}"] = _mesh_shape(mjm, mjd, g, f"geom{g}", bsdf_dict, _emitter(mjm, rc, g))
      continue
    elif int(gtype) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
      caps = "sphere" if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else "disk"
      for k, part in enumerate(_tube(pos, mat, float(size[0]), float(size[1]), caps)):
        part["bsdf"] = bsdf_dict
        scene[f"geom{g}_{k}"] = part
      continue
    else:
      raise NotImplementedError(f"geom type {gtype} is not exported to Mitsuba")
    shape["bsdf"] = bsdf_dict
    emitter = _emitter(mjm, rc, g)
    if emitter is not None:
      shape["emitter"] = emitter
    scene[f"geom{g}"] = shape

  intensity = rc.light_intensity.numpy()[0]
  for i in range(mjm.nlight):
    if not mjm.light_active[i]:
      continue
    color = (mjm.light_diffuse[i] * intensity[i]).tolist()
    if mjm.light_type[i] == pbr_lights.LIGHT_DIRECTIONAL:
      scene[f"light{i}"] = {
        "type": "directional",
        "direction": mjd.light_xdir[i].tolist(),
        "irradiance": {"type": "rgb", "value": color},
      }
    elif mjm.light_type[i] == pbr_lights.LIGHT_POINT:
      scene[f"light{i}"] = {
        "type": "point",
        "position": mjd.light_xpos[i].tolist(),
        "intensity": {"type": "rgb", "value": color},
      }
    elif mjm.light_type[i] == pbr_lights.LIGHT_SPOT:
      # Mitsuba blends from beam_width to cutoff_angle, as the renderer's smooth spot cone.
      cutoff = float(mjm.light_cutoff[i])
      origin = mjd.light_xpos[i]
      scene[f"light{i}"] = {
        "type": "spot",
        "to_world": mi.ScalarTransform4f().look_at(
          origin=origin.tolist(), target=(origin + mjd.light_xdir[i]).tolist(), up=_frame_z(mjd.light_xdir[i])[:, 0].tolist()
        ),
        "intensity": {"type": "rgb", "value": color},
        "cutoff_angle": cutoff,
        "beam_width": cutoff * pbr_lights.SPOT_INNER_FRACTION,
      }
    else:
      raise NotImplementedError("only directional, point and spot lights are exported to Mitsuba")

  ambient = rc.ambient_radiance.numpy()[0]
  if np.any(ambient > 0):
    scene["ambient"] = {"type": "constant", "radiance": {"type": "rgb", "value": ambient.tolist()}}
  if envmap is not None and rc.has_env:
    scale = float(rc.env_intensity.numpy()[0])
    image = np.ascontiguousarray(envmap[..., :3], dtype=np.float32)
    if env_to_world is None:
      env_to_world = environment_to_world(float(rc.env_yaw.numpy()[0]))
    scene["env"] = {"type": "envmap", "bitmap": mi.Bitmap(image), "scale": scale, "to_world": env_to_world}
  return scene


def environment_to_world(yaw: float) -> "mi.ScalarTransform4f":
  """Maps Mitsuba's Y-up lat-long frame onto ours (row 0 = +z, centre column = +x, yaw about z).

  Mitsuba looks up u = atan2(x, -z) / 2π, v = acos(y) / π in its local frame, so local
  (x, y, z) = (-world_y, world_z, world_x) before the yaw rotation.
  """
  frame = mi.ScalarTransform4f([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
  return mi.ScalarTransform4f().rotate([0, 0, 1], float(np.degrees(yaw))) @ frame


def render(scene: dict, seed: int = 0) -> np.ndarray:
  """Linear radiance (height, width, 3)."""
  return np.array(mi.render(mi.load_dict(scene), seed=seed))
