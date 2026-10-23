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

"""Appearance model of a Newton scene as a MuJoCo model.

Every visible shape of one world (plus the global shapes, e.g. the ground) becomes a static
MuJoCo geom with its size, mesh, colour, texture and scalar metallic / roughness. Poses are not
taken from the MuJoCo model: they are written into MuJoCo Warp's `geom_xpos` / `geom_xmat` from
Newton's body transforms every frame, so any Newton scene (USD, URDF, MJCF or built in code) can
be rendered without MuJoCo kinematics.
"""

import dataclasses
import warnings
from typing import Any, Literal, NamedTuple, Optional, Sequence

import mujoco
import newton
import numpy as np
from newton import GeoType

# Shape types rendered as MuJoCo primitives; meshes, convex meshes, cones and height fields
# become MuJoCo meshes.
_PRIMITIVES = {
  int(GeoType.PLANE): mujoco.mjtGeom.mjGEOM_PLANE,
  int(GeoType.SPHERE): mujoco.mjtGeom.mjGEOM_SPHERE,
  int(GeoType.CAPSULE): mujoco.mjtGeom.mjGEOM_CAPSULE,
  int(GeoType.ELLIPSOID): mujoco.mjtGeom.mjGEOM_ELLIPSOID,
  int(GeoType.CYLINDER): mujoco.mjtGeom.mjGEOM_CYLINDER,
  int(GeoType.BOX): mujoco.mjtGeom.mjGEOM_BOX,
}
_MESHES = {int(GeoType.MESH), int(GeoType.CONVEX_MESH), int(GeoType.CONE)}
# Newton shape colours are display (sRGB) values; MuJoCo rgba is linear for this renderer.
_DEFAULT_COLOR = (0.7, 0.7, 0.7)
# Segments of the meshed cone.
_CONE_SEGMENTS = 32

LightType = Literal["directional", "point", "spot"]
_LIGHT_NAMES: tuple[LightType, ...] = ("directional", "point", "spot")
_LIGHT_TYPES: dict[str, Any] = {
  "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  "point": mujoco.mjtLightType.mjLIGHT_POINT,
  "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
}

# (ambient, diffuse, specular) RGB of MuJoCo's headlight.
Headlight = tuple[Sequence[float], Sequence[float], Sequence[float]]


def srgb_to_linear(c: Any) -> np.ndarray:
  c = np.asarray(c, dtype=np.float64)
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@dataclasses.dataclass
class Light:
  """A light added to the scene (Newton models carry no lights).

  Attributes:
    type: "directional", "point" or "spot".
    pos: position [m] (point / spot).
    dir: direction the light travels (directional / spot).
    intensity: lux (directional) or candela (point / spot); 0 uses the renderer's calibrated
      fallback for MuJoCo-style scenes.
    color: linear RGB colour.
    range: falloff radius [m] (point / spot).
    cutoff: spot cone half-angle [deg].
    bulb_radius: emitter radius [m] for soft shadows.
    cast_shadow: whether the light casts ray-traced shadows.
  """

  type: LightType = "directional"
  pos: tuple = (0.0, 0.0, 3.0)
  dir: tuple = (-0.3, 0.4, -1.0)
  intensity: float = 0.0
  color: tuple = (1.0, 1.0, 1.0)
  range: float = 100.0
  cutoff: float = 45.0
  bulb_radius: float = 0.0
  cast_shadow: bool = True

  def __post_init__(self):
    if self.type not in _LIGHT_TYPES:
      raise ValueError(f"Light.type must be one of {list(_LIGHT_TYPES)}, got {self.type!r}")


def lights_from_mujoco(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> list[Light]:
  """The active lights of a MuJoCo model at state `mjd`.

  Keeps the lighting of an MJCF scene loaded into Newton, whose importer drops lights.
  """
  kinds: dict[int, LightType] = {int(_LIGHT_TYPES[name]): name for name in _LIGHT_NAMES}
  lights = []
  for i in range(mjm.nlight):
    if not mjm.light_active[i] or int(mjm.light_type[i]) not in kinds:
      continue
    lights.append(
      Light(
        type=kinds[int(mjm.light_type[i])],
        pos=tuple(mjd.light_xpos[i]),
        dir=tuple(mjd.light_xdir[i]),
        intensity=float(mjm.light_intensity[i]),
        color=tuple(mjm.light_diffuse[i]),
        range=float(mjm.light_range[i]),
        cutoff=float(mjm.light_cutoff[i]),
        bulb_radius=float(mjm.light_bulbradius[i]),
        cast_shadow=bool(mjm.light_castshadow[i]),
      )
    )
  return lights


def default_lights() -> list[Light]:
  """A key light with shadows and a dim fill, for scenes without an environment map."""
  return [
    Light("directional", dir=(-0.4, 0.5, -0.75), color=(1.0, 0.97, 0.92)),
    Light("directional", dir=(0.5, -0.3, -0.6), color=(0.25, 0.27, 0.3), cast_shadow=False),
  ]


@dataclasses.dataclass
class SceneMapping:
  """MuJoCo model of the scene plus the shape behind every MuJoCo geom.

  Attributes:
    mjm: compiled MuJoCo model (one world)
    geom_shape: Newton shape index of geom g in world w                    (nworld, ngeom)
    geom_offset_pos: pose of the MuJoCo geom frame in the Newton shape frame (ngeom, 3)
    geom_offset_quat: (w, x, y, z)                                             (ngeom, 4)
  """

  mjm: mujoco.MjModel
  geom_shape: np.ndarray
  geom_offset_pos: np.ndarray
  geom_offset_quat: np.ndarray


class _ShapeMesh(NamedTuple):
  """Triangle mesh of a mesh-like shape (scaled)."""

  verts: np.ndarray
  faces: np.ndarray
  uvs: Optional[np.ndarray]
  normals: Optional[np.ndarray]
  mesh: Optional[newton.Mesh]  # the Newton mesh (None for a cone)


def _shape_mesh(model: newton.Model, shape: int, gtype: int, scale: np.ndarray) -> _ShapeMesh:
  if gtype == int(GeoType.CONE):
    radius, half_height = float(scale[0]), float(scale[1])
    n = _CONE_SEGMENTS
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, -half_height)], axis=1)
    verts = np.concatenate([ring, [[0.0, 0.0, half_height], [0.0, 0.0, -half_height]]])
    faces = [[i, (i + 1) % n, n] for i in range(n)] + [[(i + 1) % n, i, n + 1] for i in range(n)]
    return _ShapeMesh(verts, np.asarray(faces), None, None, None)
  mesh = model.shape_source[shape]
  verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
  faces = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
  uvs = None if mesh.uvs is None else np.asarray(mesh.uvs, dtype=np.float64)
  normals = None if mesh.normals is None else np.asarray(mesh.normals, dtype=np.float64)
  return _ShapeMesh(verts, faces, uvs, normals, mesh)


def _add_texture(spec: mujoco.MjSpec, name: str, texture: Any) -> Optional[str]:
  """Adds a Newton mesh texture (file path or image array) as a MuJoCo 2D texture."""
  if texture is None:
    return None
  if isinstance(texture, str):
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=texture)
    return name
  image = np.asarray(texture)
  if image.dtype != np.uint8:
    image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
  if image.ndim == 2:
    image = np.repeat(image[..., None], 3, axis=-1)
  image = image[..., :3]
  tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, nchannel=3)
  tex.width, tex.height = image.shape[1], image.shape[0]
  tex.data = image.tobytes()
  return name


def _same_source(a: Any, b: Any) -> bool:
  if a is b:
    return True
  if a is None or b is None:
    return False
  return np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices)) and np.array_equal(
    np.asarray(a.indices), np.asarray(b.indices)
  )


def _check_worlds(model: newton.Model, per_world: list[np.ndarray], shape_type: np.ndarray, shape_scale: np.ndarray):
  """Raises unless every world has the same visible shapes as world 0 (type, size, mesh)."""
  first = per_world[0]
  for w, shapes in enumerate(per_world[1:], start=1):
    if len(shapes) != len(first):
      raise ValueError(
        f"newton_pbr needs every world to have the same visible shapes: world {w} has {len(shapes)}, world 0 {len(first)}"
      )
    for a, b in zip(first, shapes):
      same = shape_type[a] == shape_type[b] and np.allclose(shape_scale[a], shape_scale[b])
      if same and int(shape_type[a]) in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
        same = _same_source(model.shape_source[a], model.shape_source[b])
      if not same:
        raise ValueError(
          f"newton_pbr needs every world to have the same visible shapes: shape {b} of world {w} differs "
          f"from shape {a} of world 0 (type, scale or mesh)"
        )


def build_scene(
  model: newton.Model, lights: Optional[list[Light]], camera_fovy: float, headlight: Optional[Headlight] = None
) -> SceneMapping:
  """Builds the MuJoCo appearance model of a Newton model (worlds must have identical shapes).

  headlight: None (off) or (ambient, diffuse, specular) RGB triples of MuJoCo's headlight.
  """
  shape_world = model.shape_world.numpy()
  shape_type = model.shape_type.numpy()
  shape_scale = model.shape_scale.numpy()
  shape_flags = model.shape_flags.numpy()
  shape_color = model.shape_color.numpy() if model.shape_color is not None else None
  shape_opacity = model.shape_opacity.numpy() if getattr(model, "shape_opacity", None) is not None else None
  nworld = max(int(model.world_count), 1)

  visible = (shape_flags & int(newton.ShapeFlags.VISIBLE)) != 0
  visible &= (shape_flags & int(newton.ShapeFlags.SITE)) == 0
  skipped = sorted({GeoType(t).name for t in shape_type[visible] if t not in _PRIMITIVES and t not in _MESHES})
  if skipped:
    warnings.warn(f"newton_pbr: shape types {skipped} are not rendered", stacklevel=3)
  renderable = visible & np.isin(shape_type, list(_PRIMITIVES) + list(_MESHES))

  global_shapes = np.flatnonzero(renderable & (shape_world < 0))
  per_world = [np.flatnonzero(renderable & (shape_world == w)) for w in range(nworld)]
  _check_worlds(model, per_world, shape_type, shape_scale)
  shapes0 = np.concatenate([per_world[0], global_shapes])

  spec = mujoco.MjSpec()
  spec.visual.headlight.active = int(headlight is not None)
  if headlight is not None:
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = headlight
  spec.compiler.texturedir = ""
  # One MuJoCo mesh per (Newton mesh, scale): shapes sharing a mesh share it.
  mesh_names: dict[tuple, str] = {}
  for i, shape in enumerate(shapes0):
    gtype = int(shape_type[shape])
    scale = shape_scale[shape].astype(np.float64)
    color = srgb_to_linear(shape_color[shape] if shape_color is not None else _DEFAULT_COLOR)
    alpha = float(shape_opacity[shape]) if shape_opacity is not None else 1.0
    geom = spec.worldbody.add_geom(name=f"shape{shape}", rgba=[*color, alpha])
    mat = spec.add_material(name=f"mat{i}", rgba=[*color, alpha])
    if gtype in _PRIMITIVES:
      geom.type = _PRIMITIVES[gtype]
      if gtype == int(GeoType.PLANE):
        # Newton's width / length are full extents (0: infinite), MuJoCo's plane size is half.
        geom.size = [0.5 * scale[0], 0.5 * scale[1], 0.1]
      else:
        geom.size = scale
    else:
      shape_mesh = _shape_mesh(model, shape, gtype, scale)
      uvs = shape_mesh.uvs
      # Per-vertex UVs only: a texture is mapped only when there is one UV per vertex.
      if uvs is not None and len(uvs) != len(shape_mesh.verts):
        uvs = None
      source = shape_mesh.mesh
      key = ("cone" if source is None else id(source), *scale.tolist())
      if key not in mesh_names:
        mesh_names[key] = f"mesh{i}"
        mesh_spec = spec.add_mesh(
          name=mesh_names[key], uservert=shape_mesh.verts.ravel().tolist(), userface=shape_mesh.faces.ravel().tolist()
        )
        normals = shape_mesh.normals
        if normals is not None and len(normals) == len(shape_mesh.verts):
          mesh_spec.usernormal = normals.ravel().tolist()
        if uvs is not None:
          mesh_spec.usertexcoord = uvs.ravel().tolist()
      geom.type = mujoco.mjtGeom.mjGEOM_MESH
      geom.meshname = mesh_names[key]
      if source is not None:
        if source.roughness is not None:
          mat.roughness = float(source.roughness)
        if source.metallic is not None:
          mat.metallic = float(source.metallic)
        tex = _add_texture(spec, f"tex{i}", source.texture) if uvs is not None else None
        if tex is not None:
          mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
          mat.rgba = [1.0, 1.0, 1.0, alpha]
          geom.rgba = [1.0, 1.0, 1.0, alpha]
    geom.material = f"mat{i}"

  for i, light in enumerate(lights if lights is not None else []):
    spec.worldbody.add_light(
      name=f"light{i}",
      type=_LIGHT_TYPES[light.type],
      pos=list(light.pos),
      dir=list(light.dir),
      diffuse=list(light.color),
      intensity=light.intensity,
      range=light.range,
      cutoff=light.cutoff,
      bulbradius=light.bulb_radius,
      castshadow=light.cast_shadow,
    )
  # Placeholder camera: its pose is written from Newton's camera transforms every frame.
  spec.worldbody.add_camera(name="cam", fovy=camera_fovy)

  mjm = spec.compile()
  # MuJoCo re-centres mesh vertices on the mesh's inertial frame; the geom pose then carries
  # that offset relative to the Newton shape frame.
  offset_pos = np.array(mjm.geom_pos, dtype=np.float64).reshape(mjm.ngeom, 3)
  offset_quat = np.array(mjm.geom_quat, dtype=np.float64).reshape(mjm.ngeom, 4)
  per_world_shapes = [np.concatenate([per_world[w], global_shapes]) for w in range(nworld)]
  return SceneMapping(mjm, np.stack(per_world_shapes).astype(np.int32), offset_pos, offset_quat)
