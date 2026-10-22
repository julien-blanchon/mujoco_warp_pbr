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

"""Convert a USD stage (.usd / .usda / .usdc) into a visual-only MJCF model.

Geometry. Every renderable UsdGeom.Mesh (instance proxies included; purpose guide / proxy and
invisible prims skipped) is triangulated, its UVs and normals are expanded to face-varying
corners, it is split by GeomSubset material binding, and world transforms, metersPerUnit and
upAxis are baked into one OBJ per (mesh, material). Cube / Sphere / Capsule / Cylinder / Plane
prims become MuJoCo primitives (non-uniformly scaled spheres become ellipsoids). Everything is
a static, non-colliding geom. USD st has its origin at the bottom left like OBJ, so UVs are
written unchanged (MuJoCo flips v on load, matching its top-left image convention).

Materials (native MuJoCo fields, as tools/gltf_to_mjcf.py):

  UsdPreviewSurface   diffuseColor -> rgba / layer rgb (rgba if opacity comes from its alpha),
                      metallic / roughness / occlusion -> one packed layer orm (R=AO, G=rough,
                      B=metal; factor x texel), normal -> layer normal, emissiveColor -> layer
                      emissive + emission, opacity -> alpha / layer opacity, ior -> specular.
                      UsdUVTexture sourceColorSpace, scale and the UsdTransform2d st transform
                      (baked into the UVs) are honoured; the UV set is the PrimvarReader varname.
  OmniPBR (MDL)       by input name: diffuse_color_constant / diffuse_texture (x diffuse_tint,
                      albedo_brightness), metallic_constant / metallic_texture, reflection_
                      roughness_constant / reflectionroughness_texture (texture influences),
                      ORM_texture (enable_ORM_texture), ao_texture, normalmap_texture,
                      emissive_color / emissive_color_texture x emissive_intensity (enable_
                      emission), opacity_constant / opacity_texture (enable_opacity), specular_
                      level, texture_scale / rotate / translate.
  other MDL           modules that wrap OmniPBR (e.g. NVIDIA's Base material library) are read
                      from their argument list, with the USD shader's inputs on top; anything else
                      falls back to a UsdPreviewSurface in the same material, then to a table of
                      base colour / metallic / roughness keyed by the material name (warning).

Texture paths are resolved by USD's resolver (relative to the authoring layer), then relative to
the MDL module, and http(s) URLs through `url_roots` (URL prefix -> local mirror directory).
Textures larger than `max_texture_size` are downscaled.

Lights (UsdLux -> MuJoCo, PHYSICAL light model). USD intensities are scaled to the renderer's
units with one calibration factor, k = (responsivity * time * ISO / 100 / fStop^2 * 2^exposure)
* 1.2 * 2^EV100. The first term is the pixel value per nit of the scene's camera exposure
(UsdGeom.Camera exposure attributes when authored, else Omniverse RTX's defaults: f/5, 1/50 s,
ISO 100), the second the inverse of the renderer's exposure at `ev100` (default: the renderer's
default EV100). A scene therefore renders at the brightness it has in Omniverse RTX with its
camera, at the renderer's default exposure. With L = intensity * 2^exposure * color (colour
temperature applied; divided by the light's area when `normalize` is set):

  DistantLight        directional, illuminance L lux (Kit's convention; `angle` ignored)
  SphereLight         point, bulbradius = radius, I = L * pi r^2 cd; spot (cutoff = cone angle)
                      with ShapingAPI
  RectLight/DiskLight spot along -Z with cutoff 70 deg (or the ShapingAPI cone angle) and
                      I = L * area cd on axis: a Lambertian emitter's on-axis intensity and,
                      with the renderer's spot falloff, about its flux; bulbradius is the radius
                      of the equal-area disc. Approximation: no area shape, a cone instead of the
                      cosine lobe, and too bright close to large emitters.
  CylinderLight       point, I = L * 2 r length cd, bulbradius = radius
  DomeLight           uniform: a flat skybox of display-relative radiance L * pixel-per-nit
                      (environment light and background; clamped to 1, the LDR limit) or, with
                      dome="ambient", unclamped headlight ambient radiance (no background);
                      textured: dropped, and the envmap / envmap_intensity to pass to the
                      renderer are printed.

PreviewSurface emissiveColor is taken as display-relative (1 = full brightness, times
`preview_emission_scale`); OmniPBR emission is in nits and uses the pixel-per-nit factor.
Cameras are written with their prim names (fovy from the horizontal aperture, which Kit fits to
the image width: exact for square images).

`joint_positions` poses articulations before baking: {joint name: radians or metres}. Each
listed UsdPhysics revolute / prismatic joint moves its body1 subtree about the joint frame;
other bodies keep their authored transforms relative to their parent body.

Not converted: subdivision (meshes are rendered as their control cage, smooth normals),
UsdPreviewSurface specular workflow / clearcoat, opacity threshold (blended instead),
OmniPBR detail maps, albedo_add / desaturation, MDL graphs, skinning, animation, point
instancers, BasisCurves / Points, cones, light filters and portal / geometry lights.

With --normalize SIZE the model is centred in x/y, rests on z = 0 and is scaled so that its
largest bounding-box extent equals SIZE metres (point / spot intensities scale by SIZE^2).
With --ground it is only translated in z to rest on z = 0.

Usage:
  python tools/usd_to_mjcf.py SCENE.usd OUT_DIR [--name NAME] [--scale S]
    [--normalize SIZE | --ground]
    [--url-root URL=DIR] [--joint NAME=VALUE] [--dome skybox|ambient|drop] [--ev100 EV]
"""

import argparse
import dataclasses
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("USD_SHADE_WARN_ON_MISSING_MATERIAL_BINDING_API", "0")
os.environ.setdefault("TF_ENV_SETTING_ALERTS_ENABLED", "0")

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdLux  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
from pxr import UsdShade  # noqa: E402

# The renderer's default camera exposure (mujoco_warp_pbr._src.io.DEFAULT_EV100).
DEFAULT_EV100 = 14.96578428466209
# Omniverse RTX's default tone-mapping camera: f/5, 1/50 s, ISO 100.
KIT_FSTOP, KIT_TIME, KIT_ISO = 5.0, 0.02, 100.0
RECT_SPOT_CUTOFF = 70.0  # degrees

# Fallback base colour / metallic / roughness for MDL materials that cannot be read, matched in
# order against the lower-case material (or module) name.
MDL_FALLBACKS = [
  ("mirror", (0.95, 0.95, 0.95), 1.0, 0.02),
  ("chrome", (0.9, 0.9, 0.9), 1.0, 0.08),
  ("gold", (1.0, 0.77, 0.34), 1.0, 0.25),
  ("copper", (0.95, 0.64, 0.54), 1.0, 0.3),
  ("brass", (0.9, 0.75, 0.45), 1.0, 0.3),
  ("alumin", (0.91, 0.92, 0.92), 1.0, 0.35),
  ("steel", (0.62, 0.62, 0.62), 1.0, 0.35),
  ("metal", (0.6, 0.6, 0.6), 1.0, 0.4),
  ("glass", (0.9, 0.95, 0.95), 0.0, 0.05),
  ("water", (0.6, 0.7, 0.75), 0.0, 0.05),
  ("plastic", (0.5, 0.5, 0.5), 0.0, 0.4),
  ("rubber", (0.06, 0.06, 0.06), 0.0, 0.85),
  ("wood", (0.45, 0.3, 0.18), 0.0, 0.6),
  ("parquet", (0.45, 0.3, 0.18), 0.0, 0.55),
  ("oak", (0.5, 0.36, 0.22), 0.0, 0.6),
  ("walnut", (0.3, 0.2, 0.13), 0.0, 0.6),
  ("marble", (0.85, 0.85, 0.82), 0.0, 0.25),
  ("granite", (0.4, 0.4, 0.4), 0.0, 0.5),
  ("stone", (0.5, 0.48, 0.45), 0.0, 0.7),
  ("tile", (0.8, 0.8, 0.78), 0.0, 0.35),
  ("concrete", (0.5, 0.5, 0.48), 0.0, 0.85),
  ("asphalt", (0.12, 0.12, 0.12), 0.0, 0.9),
  ("brick", (0.5, 0.25, 0.18), 0.0, 0.85),
  ("plaster", (0.8, 0.8, 0.78), 0.0, 0.9),
  ("paint", (0.7, 0.7, 0.7), 0.0, 0.5),
  ("fabric", (0.4, 0.4, 0.4), 0.0, 0.95),
  ("cloth", (0.4, 0.4, 0.4), 0.0, 0.95),
  ("carpet", (0.35, 0.33, 0.3), 0.0, 1.0),
  ("leather", (0.25, 0.15, 0.1), 0.0, 0.6),
  ("paper", (0.85, 0.85, 0.82), 0.0, 0.9),
  ("cardboard", (0.6, 0.45, 0.3), 0.0, 0.9),
  ("ceramic", (0.9, 0.9, 0.88), 0.0, 0.2),
  ("porcelain", (0.92, 0.92, 0.9), 0.0, 0.15),
  ("black", (0.03, 0.03, 0.03), 0.0, 0.5),
  ("white", (0.85, 0.85, 0.85), 0.0, 0.5),
]
MDL_DEFAULT = ((0.5, 0.5, 0.5), 0.0, 0.5)


def _srgb_to_linear(c):
  c = np.asarray(c, dtype=np.float64)
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
  c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
  return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _fmt(values) -> str:
  return " ".join(f"{float(v):.6g}" for v in values)


def _sanitize(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_]", "_", name) or "x"


@dataclasses.dataclass(frozen=True)
class _Tex:
  """One texture channel source: local file, channel ('rgb', 'rgba', 'r', 'g', 'b', 'a'),
  colour space ('sRGB' or 'raw') and UsdUVTexture scale / bias."""

  path: str
  channel: str = "rgb"
  colorspace: str = "raw"
  scale: tuple = (1.0, 1.0, 1.0, 1.0)
  bias: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class _Mat:
  """A material reduced to the renderer's model (linear colours)."""

  base: tuple = (0.18, 0.18, 0.18)
  alpha: float = 1.0
  metallic: float = 0.0
  roughness: float = 0.5
  specular: float | None = None
  emissive: tuple = (0.0, 0.0, 0.0)  # display-relative linear
  base_tex: _Tex | None = None
  opacity_tex: _Tex | None = None  # alpha channel source (may be base_tex's alpha)
  normal_tex: _Tex | None = None
  occlusion_tex: _Tex | None = None
  roughness_tex: _Tex | None = None
  metallic_tex: _Tex | None = None
  orm_tex: _Tex | None = None  # packed R=AO, G=rough, B=metal
  emissive_tex: _Tex | None = None
  uv_name: str = "st"
  uv_xform: tuple | None = None  # (sx, sy, rotation deg, tx, ty), applied as T(R(S uv))

  def textured(self) -> bool:
    return any(
      t is not None
      for t in (
        self.base_tex,
        self.opacity_tex,
        self.normal_tex,
        self.occlusion_tex,
        self.roughness_tex,
        self.metallic_tex,
        self.orm_tex,
        self.emissive_tex,
      )
    )


# ------------------------------------------------------------------------------------------------
# MDL argument lists (OmniPBR wrappers).


def _split_args(text: str) -> list[str]:
  parts, depth, start, quote = [], 0, 0, False
  for i, ch in enumerate(text):
    if ch == '"':
      quote = not quote
    elif quote:
      continue
    elif ch in "([{":
      depth += 1
    elif ch in ")]}":
      depth -= 1
    elif ch == "," and depth == 0:
      parts.append(text[start:i])
      start = i + 1
  parts.append(text[start:])
  return [p.strip() for p in parts if p.strip()]


def _mdl_value(text: str):
  text = re.sub(r"/\*.*?\*/", "", text).strip()
  if text in ("true", "false"):
    return text == "true"
  m = re.fullmatch(r"texture_2d\((.*)\)", text, re.S)
  if m:
    args = _split_args(m.group(1))
    if not args:
      return None
    path = args[0].strip().strip('"')
    gamma = args[1] if len(args) > 1 else ""
    return ("texture", path, "sRGB" if "srgb" in gamma else "raw")
  m = re.fullmatch(r"(color|float[234]?)\((.*)\)", text, re.S)
  if m:
    values = [float(v.rstrip("f")) for v in _split_args(m.group(2))]
    size = {"color": 3, "float": 1, "float2": 2, "float3": 3, "float4": 4}[m.group(1)]
    values = values * size if len(values) == 1 else values
    return tuple(values) if size > 1 else values[0]
  try:
    return float(text.rstrip("f"))
  except ValueError:
    return None


def _read_mdl_omnipbr(path: Path, sub_identifier: str) -> dict | None:
  """Named arguments of an MDL material defined as a call of ::OmniPBR::OmniPBR*(...)."""
  try:
    text = path.read_text(errors="replace")
  except OSError:
    return None
  text = re.sub(r"//[^\n]*", "", text)
  ident = re.escape(sub_identifier) if sub_identifier else r"\w+"
  pattern = r"export\s+material\s+" + ident + r"\s*\([^)]*\)\s*(?:\[\[.*?\]\]\s*)?=\s*"
  m = re.search(pattern + r"(?:::)?OmniPBR(?:_\w+)?::OmniPBR\w*\s*\(", text, re.S)
  if not m:
    return None
  depth, i = 1, m.end()
  while i < len(text) and depth:
    depth += {"(": 1, ")": -1}.get(text[i], 0)
    i += 1
  args = {}
  for part in _split_args(text[m.end() : i - 1]):
    if ":" in part:
      key, value = part.split(":", 1)
      args[key.strip()] = _mdl_value(value)
  return args


# ------------------------------------------------------------------------------------------------
# Shading networks.


def _resolve_input(shader: UsdShade.Shader, name: str):
  """(value, (source shader, output name)) of a shader input, following connections through
  material interface inputs."""
  inp = shader.GetInput(name)
  if not inp:
    return None, None
  attrs = inp.GetValueProducingAttributes()
  if not attrs:
    return inp.Get(), None
  attr = attrs[0]
  if UsdShade.Utils.GetType(attr.GetName()) == UsdShade.AttributeType.Output:
    return None, (UsdShade.Shader(attr.GetPrim()), UsdShade.Utils.GetBaseNameAndType(attr.GetName())[0])
  return attr.Get(), None


def _shader_id(shader: UsdShade.Shader) -> str:
  ident = shader.GetShaderId()
  if ident:
    return str(ident)
  asset = shader.GetSourceAsset("mdl")
  if asset:
    return "mdl:" + (shader.GetSourceAssetSubIdentifier("mdl") or Path(asset.path).stem)
  return ""


class _Materials:
  """USD material -> _Mat, with texture path resolution."""

  def __init__(self, url_roots: dict[str, str], preview_emission_scale: float, pixel_per_nit: float):
    self.url_roots = url_roots
    self.preview_emission_scale = preview_emission_scale
    self.pixel_per_nit = pixel_per_nit
    self.cache = {}
    self.warned = set()

  def warn(self, message: str):
    if message not in self.warned:
      self.warned.add(message)
      print(f"  warning: {message}")

  def resolve(self, asset, anchors=()) -> str | None:
    """Local file for an Sdf.AssetPath / path string."""
    if asset is None:
      return None
    resolved = getattr(asset, "resolvedPath", "") or ""
    path = getattr(asset, "path", asset) or ""
    if not path:
      return None
    if resolved and Path(resolved).is_file():
      return resolved
    for prefix, root in self.url_roots.items():
      if path.startswith(prefix):
        local = Path(root) / path[len(prefix) :]
        if local.is_file():
          return str(local)
    if "://" not in path:
      for anchor in anchors:
        local = Path(anchor) / path
        if local.is_file():
          return str(local)
      if Path(path).is_file():
        return path
    self.warn(f"asset not found: {path}")
    return None

  def _mdl_file(self, shader: UsdShade.Shader) -> Path | None:
    asset = shader.GetSourceAsset("mdl")
    if not asset:
      return None
    local = self.resolve(asset)
    return Path(local) if local else None

  def get(self, material: UsdShade.Material | None) -> _Mat | None:
    if material is None or not material.GetPrim().IsValid():
      return None
    key = material.GetPath()
    if key not in self.cache:
      self.cache[key] = self._convert(material)
    return self.cache[key]

  def _convert(self, material: UsdShade.Material) -> _Mat:
    name = material.GetPrim().GetName()
    mdl_shader = material.ComputeSurfaceSource("mdl")[0]
    preview = material.ComputeSurfaceSource("")[0]
    if preview and _shader_id(preview) != "UsdPreviewSurface":
      preview = None
    if not preview:
      # Some assets put the preview surface on a non-standard output.
      for output in material.GetOutputs():
        for source in output.GetConnectedSources()[0]:
          shader = UsdShade.Shader(source.source.GetPrim())
          if shader and _shader_id(shader) == "UsdPreviewSurface":
            preview = shader
    if mdl_shader and _shader_id(mdl_shader) == "UsdPreviewSurface":
      preview, mdl_shader = mdl_shader, None

    if mdl_shader:
      sub = mdl_shader.GetSourceAssetSubIdentifier("mdl") or ""
      asset = mdl_shader.GetSourceAsset("mdl")
      module_name = Path(asset.path).stem if asset else sub
      defaults, module = None, None
      if module_name.startswith("OmniPBR") or sub.startswith("OmniPBR"):
        defaults = {}  # Omniverse core library module: the USD inputs are the arguments.
      elif (module := self._mdl_file(mdl_shader)) is not None:
        defaults = _read_mdl_omnipbr(module, sub)
      if defaults is not None:
        return self._omnipbr(mdl_shader, defaults, module.parent if module else None)
    if preview:
      return self._preview(preview)
    label = f"{name} ({_shader_id(mdl_shader)})" if mdl_shader else name
    lower = (name + " " + (_shader_id(mdl_shader) if mdl_shader else "")).lower()
    base, metallic, roughness = next(((c, m, r) for k, c, m, r in MDL_FALLBACKS if k in lower), MDL_DEFAULT)
    self.warn(f"material {label}: MDL not readable, using fallback {base} metallic {metallic} roughness {roughness}")
    return _Mat(base=tuple(_srgb_to_linear(base)), metallic=metallic, roughness=roughness)

  # UsdPreviewSurface ---------------------------------------------------------------------------

  def _texture(self, shader: UsdShade.Shader, output: str, colour: bool):
    """(_Tex, uv name, uv transform) of a UsdUVTexture output."""
    if _shader_id(shader) != "UsdUVTexture":
      self.warn(f"unsupported shading node {_shader_id(shader)} at {shader.GetPath()}")
      return None, None, None
    file, _ = _resolve_input(shader, "file")
    path = self.resolve(file, [str(Path(file.resolvedPath).parent)] if file and file.resolvedPath else [])
    if path is None:
      return None, None, None
    space, _ = _resolve_input(shader, "sourceColorSpace")
    space = str(space or "auto")
    colorspace = "sRGB" if space == "sRGB" or (space == "auto" and colour) else "raw"
    scale, _ = _resolve_input(shader, "scale")
    bias, _ = _resolve_input(shader, "bias")
    scale = tuple(scale) if scale is not None else (1.0,) * 4
    bias = tuple(bias) if bias is not None else (0.0,) * 4
    uv_name, uv_xform = "st", None
    _, st = _resolve_input(shader, "st")
    while st is not None:
      node = st[0]
      ident = _shader_id(node)
      if ident == "UsdTransform2d":
        s, _ = _resolve_input(node, "scale")
        r, _ = _resolve_input(node, "rotation")
        t, _ = _resolve_input(node, "translation")
        s = tuple(s) if s is not None else (1.0, 1.0)
        t = tuple(t) if t is not None else (0.0, 0.0)
        uv_xform = (s[0], s[1], float(r or 0.0), t[0], t[1])
        _, st = _resolve_input(node, "in")
      elif ident.startswith("UsdPrimvarReader"):
        var, _ = _resolve_input(node, "varname")
        uv_name = str(var) if var else "st"
        st = None
      else:
        self.warn(f"unsupported st node {ident} at {node.GetPath()}")
        st = None
    return _Tex(path, output, colorspace, scale, bias), uv_name, uv_xform

  def _preview(self, shader: UsdShade.Shader) -> _Mat:
    fields = {}
    uv = {}

    def value(name, default):
      v, src = _resolve_input(shader, name)
      return default if v is None else v, src

    def tex(src, colour):
      t, uv_name, uv_xform = self._texture(src[0], src[1], colour)
      if t is not None:
        uv.setdefault("name", uv_name)
        uv.setdefault("xform", uv_xform)
      return t

    diffuse, src = value("diffuseColor", (0.18, 0.18, 0.18))
    fields["base"] = tuple(float(c) for c in diffuse)
    if src:
      t = tex(src, True)
      if t is not None:
        fields["base_tex"] = t
        fields["base"] = tuple(t.scale[:3])
        if any(t.bias[:3]):
          self.warn(f"{shader.GetPath()}: diffuse texture bias ignored")
    opacity, src = value("opacity", 1.0)
    fields["alpha"] = float(opacity)
    if src:
      t = tex(src, False)
      if t is not None:
        fields["opacity_tex"] = t
        fields["alpha"] = t.scale[3 if t.channel == "a" else "rgba".index(t.channel[0])]
    threshold, _ = value("opacityThreshold", 0.0)
    if float(threshold) > 0.0:
      self.warn(f"{shader.GetPath()}: opacityThreshold (cutout) rendered as blended alpha")
    for name, field, default in (("metallic", "metallic", 0.0), ("roughness", "roughness", 0.5)):
      v, src = value(name, default)
      fields[field] = float(v)
      if src:
        t = tex(src, False)
        if t is not None:
          fields[f"{field}_tex"] = t
          fields[field] = 1.0
    occlusion, src = value("occlusion", 1.0)
    if src:
      t = tex(src, False)
      if t is not None:
        fields["occlusion_tex"] = t
    _, src = value("normal", None)
    if src:
      t = tex(src, False)
      if t is not None:
        fields["normal_tex"] = t
    emissive, src = value("emissiveColor", (0.0, 0.0, 0.0))
    fields["emissive"] = tuple(float(c) * self.preview_emission_scale for c in emissive)
    if src:
      t = tex(src, True)
      if t is not None:
        fields["emissive_tex"] = t
        fields["emissive"] = tuple(float(c) * self.preview_emission_scale for c in t.scale[:3])
    workflow, _ = value("useSpecularWorkflow", 0)
    if int(workflow):
      self.warn(f"{shader.GetPath()}: specular workflow approximated as metallic 0")
      fields["metallic"] = 0.0
      fields.pop("metallic_tex", None)
    ior, _ = _resolve_input(shader, "ior")
    if ior is not None and shader.GetInput("ior").GetAttr().HasAuthoredValue():
      f0 = ((float(ior) - 1.0) / (float(ior) + 1.0)) ** 2
      fields["specular"] = min(math.sqrt(f0 / 0.16), 1.0)
    fields["uv_name"] = uv.get("name") or "st"
    fields["uv_xform"] = uv.get("xform")
    return _Mat(**fields)

  # OmniPBR -------------------------------------------------------------------------------------

  def _omnipbr(self, shader: UsdShade.Shader, defaults: dict, module_dir: Path | None) -> _Mat:
    anchors = [str(module_dir)] if module_dir else []

    def get(name, default):
      inp = shader.GetInput(name)
      if inp:
        v, src = _resolve_input(shader, name)
        if src is not None:
          self.warn(f"{shader.GetPath()}: connected OmniPBR input {name} ignored")
        elif v is not None and (inp.GetAttr().HasAuthoredValue() or not isinstance(v, Sdf.AssetPath)):
          if isinstance(v, Sdf.AssetPath):
            if not v.path:
              return None
            local = self.resolve(v, anchors)
            space = inp.GetAttr().GetColorSpace() or ""
            return ("texture", local, space) if local else None
          return v
      v = defaults.get(name)
      if isinstance(v, tuple) and len(v) == 3 and v[0] == "texture":
        local = self.resolve(v[1], anchors)
        return ("texture", local, v[2]) if local else None
      return default if v is None else v

    def texture(name, channel, colour):
      v = get(name, None)
      if not (isinstance(v, tuple) and v and v[0] == "texture" and v[1]):
        return None
      space = v[2] if v[2] in ("sRGB", "raw") else ("sRGB" if colour else "raw")
      return _Tex(v[1], channel, space)

    fields = {}
    tint = np.array(get("diffuse_tint", (1.0, 1.0, 1.0)), dtype=np.float64)
    brightness = float(get("albedo_brightness", 1.0))
    base_tex = texture("diffuse_texture", "rgb", True)
    if base_tex is not None:
      fields["base_tex"] = base_tex
      fields["base"] = tuple(tint * brightness)
    else:
      fields["base"] = tuple(np.array(get("diffuse_color_constant", (0.2, 0.2, 0.2)), dtype=np.float64) * tint * brightness)

    rough = float(get("reflection_roughness_constant", 0.5))
    metal = float(get("metallic_constant", 0.0))
    if bool(get("enable_ORM_texture", False)) and texture("ORM_texture", "rgb", False) is not None:
      fields["orm_tex"] = texture("ORM_texture", "rgb", False)
      rough, metal = 1.0, 1.0
    else:
      rt = texture("reflectionroughness_texture", "r", False)
      if rt is not None and float(get("reflection_roughness_texture_influence", 0.0)) > 0.5:
        fields["roughness_tex"], rough = rt, 1.0
      mt = texture("metallic_texture", "r", False)
      if mt is not None and float(get("metallic_texture_influence", 0.0)) > 0.5:
        fields["metallic_tex"], metal = mt, 1.0
      ao = texture("ao_texture", "r", False)
      if ao is not None:
        fields["occlusion_tex"] = ao
    fields["roughness"], fields["metallic"] = rough, metal
    fields["specular"] = float(get("specular_level", 0.5))
    normal = texture("normalmap_texture", "rgb", False)
    if normal is not None:
      fields["normal_tex"] = normal
    if bool(get("enable_emission", False)):
      intensity = float(get("emissive_intensity", 40.0)) * self.pixel_per_nit
      colour = np.array(get("emissive_color", (1.0, 0.1, 0.1)), dtype=np.float64)
      et = texture("emissive_color_texture", "rgb", True)
      if et is not None:
        fields["emissive_tex"] = et
        colour = np.ones(3)
      fields["emissive"] = tuple(colour * intensity)
    if bool(get("enable_opacity", False)):
      fields["alpha"] = float(get("opacity_constant", 1.0))
      ot = texture("opacity_texture", "r", False)
      if ot is not None:
        fields["opacity_tex"] = ot
    for name in ("albedo_add", "albedo_desaturation"):
      if abs(float(get(name, 0.0))) > 1e-3:
        self.warn(f"{shader.GetPath()}: OmniPBR {name} ignored")
    if bool(get("project_uvw", False)):
      self.warn(f"{shader.GetPath()}: OmniPBR project_uvw ignored (mesh UVs used)")
    scale = tuple(get("texture_scale", (1.0, 1.0)))
    rotate = float(get("texture_rotate", 0.0))
    translate = tuple(get("texture_translate", (0.0, 0.0)))
    if scale != (1.0, 1.0) or rotate != 0.0 or translate != (0.0, 0.0):
      fields["uv_xform"] = (scale[0], scale[1], rotate, translate[0], translate[1])
    fields["uv_name"] = "st"
    return _Mat(**fields)


# ------------------------------------------------------------------------------------------------
# Images.


class _Textures:
  """Writes MJCF textures (PNG) from _Tex sources, deduplicated."""

  def __init__(self, out_dir: Path, max_size: int):
    self.out_dir = out_dir
    self.max_size = max_size
    self.images = {}
    self.written = {}  # key -> texture name
    self.entries = []  # (name, file, colorspace)

  def image(self, path: str) -> np.ndarray:
    """RGBA uint8 image (downscaled to max_size)."""
    if path not in self.images:
      img = Image.open(path)
      if img.mode in ("I;16", "I;16B", "I"):
        img = Image.fromarray((np.asarray(img, dtype=np.float64) / 257.0).clip(0, 255).astype(np.uint8))
      img = img.convert("RGBA")
      if max(img.size) > self.max_size:
        factor = self.max_size / max(img.size)
        img = img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))), Image.LANCZOS)
      self.images[path] = np.asarray(img)
    return self.images[path]

  def channel(self, tex: _Tex, shape=None) -> np.ndarray:
    """One channel as float in [0, 1] (scale and bias applied), resized to shape."""
    img = self.image(tex.path)
    index = "rgba".index(tex.channel[0]) if tex.channel != "rgb" else 0
    c = img[..., index].astype(np.float64) / 255.0
    c = c * tex.scale[index] + tex.bias[index]
    if shape is not None and c.shape != shape:
      c = (
        np.asarray(Image.fromarray((c.clip(0, 1) * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BILINEAR)) / 255.0
      )
    return c

  def _save(self, key, stem: str, data: np.ndarray, colorspace: str) -> str:
    if key in self.written:
      return self.written[key]
    name = f"t{len(self.entries)}_{_sanitize(stem)[:40]}"
    file = f"textures/{name}.png"
    Image.fromarray(data).save(self.out_dir / file)
    self.entries.append((name, file, "sRGB" if colorspace == "sRGB" else "linear"))
    self.written[key] = name
    return name

  def base(self, mat: _Mat) -> tuple[str, str] | None:
    """(texture, role) of the base colour layer."""
    tex, alpha = mat.base_tex, mat.opacity_tex
    if tex is None:
      return None
    rgba = alpha is not None and alpha.path == tex.path and alpha.channel == "a"
    key = ("base", tex.path, tex.colorspace, rgba)
    img = self.image(tex.path)
    data = img if rgba else np.ascontiguousarray(img[..., :3])
    return self._save(key, Path(tex.path).stem, data, tex.colorspace), "rgba" if rgba else "rgb"

  def opacity(self, mat: _Mat) -> str | None:
    tex = mat.opacity_tex
    if tex is None or (mat.base_tex is not None and tex.path == mat.base_tex.path and tex.channel == "a"):
      return None
    c = (self.channel(dataclasses.replace(tex, scale=(1.0,) * 4, bias=(0.0,) * 4)) * 255 + 0.5).astype(np.uint8)
    return self._save(("opacity", tex.path, tex.channel), Path(tex.path).stem + "_opacity", np.stack([c] * 3, -1), "raw")

  def normal(self, mat: _Mat) -> str | None:
    tex = mat.normal_tex
    if tex is None:
      return None
    data = np.ascontiguousarray(self.image(tex.path)[..., :3])
    return self._save(("normal", tex.path), Path(tex.path).stem, data, "raw")

  def orm(self, mat: _Mat) -> str | None:
    if mat.orm_tex is not None:
      t = mat.orm_tex
      data = np.ascontiguousarray(self.image(t.path)[..., :3])
      return self._save(("orm", t.path), Path(t.path).stem, data, "raw")
    sources = [mat.occlusion_tex, mat.roughness_tex, mat.metallic_tex]
    if all(s is None for s in sources):
      return None
    shape = max((self.image(s.path).shape[:2] for s in sources if s is not None), key=lambda s: s[0] * s[1])
    data = np.full((*shape, 3), 255, dtype=np.uint8)
    for i, s in enumerate(sources):
      if s is not None:
        data[..., i] = (self.channel(s, shape).clip(0, 1) * 255 + 0.5).astype(np.uint8)
    key = ("orm", *(dataclasses.astuple(s) if s else None for s in sources))
    stem = Path(next(s for s in sources if s is not None).path).stem
    return self._save(key, stem + "_orm", data, "raw")

  def emissive(self, mat: _Mat) -> tuple[str, float] | None:
    """(texture, emission) for the emissive layer; colour factors are baked into the texture."""
    colour = np.array(mat.emissive, dtype=np.float64)
    peak = float(colour.max())
    if peak <= 0.0:
      return None
    factor = colour / peak
    if mat.emissive_tex is not None:
      t = mat.emissive_tex
      img = self.image(t.path)[..., :3].astype(np.float64) / 255.0
      lin = _srgb_to_linear(img) if t.colorspace == "sRGB" else img
      data = (_linear_to_srgb(lin * factor) * 255 + 0.5).astype(np.uint8)
      key = ("emissive", t.path, tuple(factor.round(4)))
      return self._save(key, Path(t.path).stem + "_emissive", data, "sRGB"), peak
    data = (_linear_to_srgb(factor) * 255 + 0.5).astype(np.uint8).reshape(1, 1, 3)
    return self._save(("emissive", tuple(factor.round(4))), "emissive", data, "sRGB"), peak


# ------------------------------------------------------------------------------------------------
# Geometry.


def _np_matrix(m: Gf.Matrix4d) -> np.ndarray:
  """Gf row-vector matrix -> 4x4 column-vector affine."""
  return np.array(m, dtype=np.float64).T


def _primvar(mesh: UsdGeom.Mesh, name: str):
  pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(name)
  if not pv or not pv.HasValue():
    return None, None
  values = pv.ComputeFlattened()
  if values is None:
    return None, None
  return np.asarray(values, dtype=np.float64), pv.GetInterpolation()


def _corner_values(values, interp, counts, indices, ncorner):
  """Expand a primvar to face-varying corners."""
  if values is None:
    return None
  if interp == UsdGeom.Tokens.faceVarying:
    return values[:ncorner] if len(values) >= ncorner else None
  if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
    return values[indices] if len(values) > indices.max(initial=0) else None
  if interp == UsdGeom.Tokens.uniform:
    return np.repeat(values, counts, axis=0) if len(values) >= len(counts) else None
  if interp == UsdGeom.Tokens.constant:
    return np.repeat(values[:1], ncorner, axis=0)
  return None


def _triangulate(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Fan triangulation: (corner index triples, face id per triangle)."""
  starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
  ntri = np.maximum(counts - 2, 0)
  face = np.repeat(np.arange(len(counts)), ntri)
  base = np.repeat(starts, ntri)
  k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
  return np.stack([base, base + k + 1, base + k + 2], axis=1), face


def _apply_uv_xform(uv: np.ndarray, xform) -> np.ndarray:
  if xform is None:
    return uv
  sx, sy, rot, tx, ty = xform
  c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
  u, v = uv[:, 0] * sx, uv[:, 1] * sy
  return np.stack([c * u - s * v + tx, s * u + c * v + ty], axis=1)


def _write_obj(path: Path, pos, normal, uv, faces):
  lines = [f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}" for p in pos]
  if uv is not None:
    lines += [f"vt {t[0]:.7g} {t[1]:.7g}" for t in uv]
  if normal is not None:
    lines += [f"vn {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}" for n in normal]
  for f in faces + 1:
    if uv is not None and normal is not None:
      lines.append(f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}")
    elif uv is not None:
      lines.append(f"f {f[0]}/{f[0]} {f[1]}/{f[1]} {f[2]}/{f[2]}")
    elif normal is not None:
      lines.append(f"f {f[0]}//{f[0]} {f[1]}//{f[1]} {f[2]}//{f[2]}")
    else:
      lines.append(f"f {f[0]} {f[1]} {f[2]}")
  path.write_text("\n".join(lines) + "\n")


def _axis_rotation(axis: str) -> np.ndarray:
  """Rotation taking MuJoCo's primitive axis (z) to the USD prim's axis."""
  if axis == "X":
    return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
  if axis == "Y":
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
  return np.eye(3)


def _mat_to_quat(r: np.ndarray) -> np.ndarray:
  w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
  x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
  y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
  z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def _orthonormalize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(rotation, per-axis scale) of a linear map without shear."""
  scale = np.linalg.norm(a, axis=0)
  u, _, vt = np.linalg.svd(a / np.maximum(scale, 1e-12))
  r = u @ vt
  if np.linalg.det(r) < 0:
    r[:, 2] *= -1
    scale[2] *= -1
  return r, np.abs(scale)


# ------------------------------------------------------------------------------------------------
# Articulation posing.


def _pose_deltas(stage: Usd.Stage, xcache: UsdGeom.XformCache, joint_positions: dict, mpu: float) -> dict:
  """Body path -> 4x4 world-space correction (column form) for the requested joint positions."""
  if not joint_positions:
    return {}
  parent_joint = {}
  predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
  for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
    if not prim.IsA(UsdPhysics.Joint):
      continue
    joint = UsdPhysics.Joint(prim)
    body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    if body0 and body1:
      parent_joint[body1[0]] = (prim, body0[0])
  unused = set(joint_positions)
  world = {}

  def old_world(path):
    return _np_matrix(xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))

  def new_world(path):
    if path in world:
      return world[path]
    if path not in parent_joint:
      world[path] = old_world(path)
      return world[path]
    prim, parent = parent_joint[path]
    rel = np.linalg.inv(old_world(parent)) @ old_world(path)  # child in parent frame
    name = prim.GetName()
    if name in joint_positions and (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
      unused.discard(name)
      joint = UsdPhysics.Joint(prim)
      pos1 = np.array(joint.GetLocalPos1Attr().Get() or (0.0, 0.0, 0.0), dtype=np.float64)
      rot1 = joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0)
      frame = np.eye(4)  # joint frame in child coordinates
      frame[:3, :3] = _np_matrix(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(rot1))))[:3, :3]
      frame[:3, 3] = pos1
      axis = np.zeros(3)
      axis["XYZ".index(str(prim.GetAttribute("physics:axis").Get() or "X"))] = 1.0
      motion = np.eye(4)
      q = float(joint_positions[name])
      if prim.IsA(UsdPhysics.RevoluteJoint):
        motion[:3, :3] = _np_matrix(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), math.degrees(q))))[:3, :3]
      else:
        motion[:3, 3] = axis * q / mpu
      rel = rel @ frame @ motion @ np.linalg.inv(frame)
    world[path] = new_world(parent) @ rel
    return world[path]

  deltas = {}
  for path in parent_joint:
    deltas[path] = new_world(path) @ np.linalg.inv(old_world(path))
  if unused:
    print(f"  warning: joints not found: {sorted(unused)}")
  return deltas


# ------------------------------------------------------------------------------------------------
# Converter.


@dataclasses.dataclass
class _Light:
  name: str
  kind: str  # directional / point / spot
  pos: np.ndarray
  dir: np.ndarray
  colour: np.ndarray
  intensity: float  # lux (directional) or candela, before normalisation scaling
  bulbradius: float = 0.0
  cutoff: float = 0.0
  castshadow: bool = True


def _light_input(prim: Usd.Prim, name: str, default):
  """UsdLux input, preferring authored `inputs:name`, then legacy un-namespaced `name`."""
  for attr_name in (f"inputs:{name}", name):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasAuthoredValue():
      return attr.Get()
  attr = prim.GetAttribute(f"inputs:{name}")
  if attr and attr.Get() is not None:
    return attr.Get()
  return default


def _blackbody(kelvin: float) -> np.ndarray:
  try:
    rgb = np.array(UsdLux.BlackbodyTemperatureAsRgb(kelvin), dtype=np.float64)
  except AttributeError:
    # Planckian locus approximation (Tanner Helland), normalised to max 1.
    t = kelvin / 100.0
    r = 1.0 if t <= 66 else 1.292936186 * (t - 60) ** -0.1332047592
    g = 0.390081579 * math.log(t) - 0.631841444 if t <= 66 else 1.129890861 * (t - 60) ** -0.0755148492
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else 0.543206789 * math.log(t - 10) - 1.196254089)
    rgb = _srgb_to_linear(np.clip([r, g, b], 0, 1))
  return rgb / max(rgb.max(), 1e-9)


class _Converter:
  def __init__(self, path: Path, out_dir: Path, name: str, options: dict):
    self.path = path
    self.out_dir = out_dir
    self.name = name
    self.opt = options
    self.stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if not self.stage:
      raise RuntimeError(f"cannot open {path}")
    for spec in options.get("variants", {}).items():
      self._select_variant(*spec)
    self.xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    self.mpu = UsdGeom.GetStageMetersPerUnit(self.stage) or 1.0
    up = UsdGeom.GetStageUpAxis(self.stage)
    # Stage -> MuJoCo (column form): units, Y-up -> Z-up ((x, y, z) -> (x, -z, y)), user scale.
    self.root = np.eye(4)
    if up == UsdGeom.Tokens.y:
      self.root[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    self.root[:3, :3] *= self.mpu * options["scale"]
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)
    self.pixel_per_nit = self._pixel_per_nit()
    self.light_scale = self.pixel_per_nit * 1.2 * 2.0 ** options["ev100"]
    self.materials = _Materials(options["url_roots"], options["preview_emission_scale"], self.pixel_per_nit)
    self.textures = _Textures(out_dir, options["max_texture_size"])
    self.mjcf_materials = {}  # _Mat -> (name, xml)
    self.meshes = []  # (name, pos, normal, uv, faces)
    self.prims = []  # (type, size, affine, material)
    self.lights = []
    self.cameras = []  # (name, affine, fovy)
    self.ambient = None
    self.skybox = None
    self.names = {}
    self.deltas = _pose_deltas(self.stage, self.xcache, options["joint_positions"], self.mpu)

  def _select_variant(self, spec: str, value: str):
    path, _, set_name = spec.rpartition(":")
    prim = self.stage.GetPrimAtPath(path) if path else self.stage.GetDefaultPrim()
    if prim and prim.GetVariantSets().HasVariantSet(set_name):
      prim.GetVariantSets().GetVariantSet(set_name).SetVariantSelection(value)
    else:
      print(f"  warning: variant set {spec} not found")

  def _unique(self, base: str) -> str:
    base = _sanitize(base)[:48]
    n = self.names.get(base, 0)
    self.names[base] = n + 1
    return base if n == 0 else f"{base}_{n}"

  def _pixel_per_nit(self) -> float:
    """Display value per nit of the scene camera (Kit's default camera if not authored)."""
    cams = [p for p in self.stage.Traverse() if p.IsA(UsdGeom.Camera)]
    wanted = self.opt.get("camera")
    cams.sort(key=lambda p: p.GetName() != wanted)
    fstop, time, iso, responsivity, ev = KIT_FSTOP, KIT_TIME, KIT_ISO, 1.0, 0.0
    for cam in cams:
      attrs = {n: cam.GetAttribute(f"exposure:{n}") for n in ("fStop", "time", "iso", "responsivity")}
      if any(a and a.HasAuthoredValue() for a in attrs.values()):
        get = lambda n, d: attrs[n].Get() if attrs[n] and attrs[n].HasAuthoredValue() else d  # noqa: E731
        fstop, time, iso, responsivity = get("fStop", fstop), get("time", time), get("iso", iso), get("responsivity", 1.0)
        exposure = cam.GetAttribute("exposure")
        ev = float(exposure.Get()) if exposure and exposure.HasAuthoredValue() else 0.0
        print(f"  exposure from camera {cam.GetPath()}: f/{fstop:g} {time:g} s ISO {iso:g} x{responsivity:g}")
        break
    return float(responsivity) * float(time) * float(iso) / 100.0 / float(fstop) ** 2 * 2.0**ev

  def world(self, prim: Usd.Prim) -> np.ndarray:
    m = _np_matrix(self.xcache.GetLocalToWorldTransform(prim))
    if self.deltas:
      p = prim
      while p and p.GetPath() != Sdf.Path.absoluteRootPath:
        if p.GetPath() in self.deltas:
          m = self.deltas[p.GetPath()] @ m
          break
        p = p.GetParent()
    return self.root @ m

  def mjcf_material(self, mat: _Mat | None) -> str | None:
    if mat is None:
      return None
    if mat in self.mjcf_materials:
      return self.mjcf_materials[mat][0]
    name = f"mat{len(self.mjcf_materials)}"
    layers = []
    base = self.textures.base(mat)
    if base:
      layers.append(base)
    normal = self.textures.normal(mat)
    if normal:
      layers.append((normal, "normal"))
    orm = self.textures.orm(mat)
    if orm:
      layers.append((orm, "orm"))
    emissive = self.textures.emissive(mat)
    if emissive:
      layers.append((emissive[0], "emissive"))
    opacity = self.textures.opacity(mat)
    if opacity:
      layers.append((opacity, "opacity"))
    rgba = [*np.clip(mat.base, 0.0, None), float(np.clip(mat.alpha, 0.0, 1.0))]
    attrs = f'name="{name}" rgba="{_fmt(rgba)}" metallic="{mat.metallic:.6g}" roughness="{mat.roughness:.6g}"'
    if mat.specular is not None:
      attrs += f' specular="{mat.specular:.6g}"'
    if emissive:
      attrs += f' emission="{emissive[1]:.6g}"'
    body = "".join(f'\n      <layer texture="{tex}" role="{role}"/>' for tex, role in layers)
    self.mjcf_materials[mat] = (name, f"    <material {attrs}>{body}\n    </material>" if body else f"    <material {attrs}/>")
    return name

  def material_of(self, prim: Usd.Prim) -> _Mat | None:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    mat = self.materials.get(material) if material else None
    if mat is None:
      colour = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
      if colour and colour.HasValue():
        values = np.asarray(colour.ComputeFlattened(), dtype=np.float64).reshape(-1, 3)
        if len(values):
          mat = _Mat(base=tuple(values.mean(axis=0)), roughness=0.5)
    return mat

  # Meshes -------------------------------------------------------------------------------------

  def mesh(self, prim: Usd.Prim):
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
      return
    points = np.asarray(points, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if counts.sum() != len(indices) or indices.max() >= len(points):
      print(f"  skipping {prim.GetPath()}: inconsistent topology")
      return
    ncorner = len(indices)
    tris, tri_face = _triangulate(counts)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
      tris = tris[:, ::-1]

    pos = points[indices]
    normals = None
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("normals")
    if pv and pv.HasValue():
      normals = _corner_values(*_primvar(mesh, "normals"), counts, indices, ncorner)
    else:
      attr = mesh.GetNormalsAttr()
      if attr.HasValue():
        normals = _corner_values(
          np.asarray(attr.Get(), dtype=np.float64), mesh.GetNormalsInterpolation(), counts, indices, ncorner
        )
    if normals is None:
      # Smooth (subdivision surface) or faceted (scheme none) normals.
      a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
      face_n = np.cross(b - a, c - a)
      if mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none:
        per_face = np.zeros((len(counts), 3))
        np.add.at(per_face, tri_face, face_n)
        normals = np.repeat(per_face, counts, axis=0)
      else:
        vn = np.zeros((len(points), 3))
        np.add.at(vn, indices[tris.reshape(-1)], np.repeat(face_n, 3, axis=0))
        normals = vn[indices]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # Per-face materials: GeomSubsets (materialBind family) over the mesh's own binding.
    binding = UsdShade.MaterialBindingAPI(prim)
    face_mat = np.zeros(len(counts), dtype=np.int64)
    mats = [self.material_of(prim)]
    for subset in binding.GetMaterialBindSubsets():
      ids = np.asarray(subset.GetIndicesAttr().Get() or [], dtype=np.int64)
      ids = ids[(ids >= 0) & (ids < len(counts))]
      if len(ids):
        mats.append(self.material_of(subset.GetPrim()))
        face_mat[ids] = len(mats) - 1

    world = self.world(prim)
    lin = world[:3, :3]
    wpos = pos @ lin.T + world[:3, 3]
    wnrm = normals @ np.linalg.inv(lin)
    wnrm /= np.maximum(np.linalg.norm(wnrm, axis=1, keepdims=True), 1e-12)
    if np.linalg.det(lin) < 0.0:
      tris = tris[:, ::-1]
    uv_cache = {}
    for mi, mat in enumerate(mats):
      sel = tris[face_mat[tri_face] == mi]
      if not len(sel):
        continue
      uv = None
      uv_name = mat.uv_name if mat is not None else "st"
      if mat is not None and mat.textured():
        if uv_name not in uv_cache:
          values, interp = _primvar(mesh, uv_name)
          if values is None and uv_name != "st":
            values, interp = _primvar(mesh, "st")
          if values is None:
            for p in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
              if p.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array) and p.HasValue():
                values, interp = _primvar(mesh, p.GetPrimvarName())
                break
          uv_cache[uv_name] = _corner_values(values, interp, counts, indices, ncorner)
        uv = uv_cache[uv_name]
        if uv is None:
          print(f"  warning: {prim.GetPath()} has no UVs for its textured material")
        else:
          uv = _apply_uv_xform(uv[:, :2], mat.uv_xform)
      corners = np.unique(sel.reshape(-1))
      data = [wpos[corners], wnrm[corners]] + ([uv[corners]] if uv is not None else [])
      keys = np.concatenate([np.round(d, 6) for d in data], axis=1)
      _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
      remap = np.empty(ncorner, dtype=np.int64)
      remap[corners] = inverse.reshape(-1)
      faces = remap[sel]
      keep = corners[first]
      name = self._unique(f"{prim.GetName()}_{mi}" if len(mats) > 1 else prim.GetName())
      self.meshes.append((name, wpos[keep], wnrm[keep], uv[keep] if uv is not None else None, faces, self.mjcf_material(mat)))

  # Primitives ---------------------------------------------------------------------------------

  def primitive(self, prim: Usd.Prim):
    world = self.world(prim)
    axis = str(prim.GetAttribute("axis").Get() or "Z") if prim.HasAttribute("axis") else "Z"
    if prim.IsA(UsdGeom.Cube):
      half = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get())
      kind, local = "box", np.array([half, half, half])
    elif prim.IsA(UsdGeom.Sphere):
      r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
      kind, local = "sphere", np.array([r, r, r])
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
      geom = UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(prim)
      r, h = float(geom.GetRadiusAttr().Get()), float(geom.GetHeightAttr().Get())
      kind, local = ("capsule" if prim.IsA(UsdGeom.Capsule) else "cylinder"), np.array([r, r, 0.5 * h])
    elif hasattr(UsdGeom, "Plane") and prim.IsA(UsdGeom.Plane):
      plane = UsdGeom.Plane(prim)
      kind, local = "plane", np.array([0.5 * float(plane.GetWidthAttr().Get()), 0.5 * float(plane.GetLengthAttr().Get()), 0.0])
    else:
      print(f"  skipping {prim.GetPath()}: {prim.GetTypeName()} is not supported")
      return
    affine = world.copy()
    affine[:3, :3] = world[:3, :3] @ _axis_rotation(axis)
    self.prims.append((self._unique(prim.GetName()), kind, local, affine, self.mjcf_material(self.material_of(prim))))

  # Lights -------------------------------------------------------------------------------------

  def light(self, prim: Usd.Prim):
    kind = prim.GetTypeName()
    world = self.world(prim)
    lin = world[:3, :3]
    colour = np.array(_light_input(prim, "color", (1.0, 1.0, 1.0)), dtype=np.float64)
    if _light_input(prim, "enableColorTemperature", False):
      colour = colour * _blackbody(float(_light_input(prim, "colorTemperature", 6500.0)))
    radiance = float(_light_input(prim, "intensity", 1.0)) * 2.0 ** float(_light_input(prim, "exposure", 0.0))
    radiance *= float(_light_input(prim, "diffuse", 1.0))
    normalize = bool(_light_input(prim, "normalize", False))
    shadow = bool(_light_input(prim, "shadow:enable", True))
    cone = _light_input(prim, "shaping:cone:angle", None)
    cone = float(cone) if cone is not None and float(cone) < 90.0 else None
    pos, direction = world[:3, 3], lin @ np.array([0.0, 0.0, -1.0])
    direction /= max(np.linalg.norm(direction), 1e-12)
    scale = np.linalg.norm(lin, axis=0)  # metres per local unit, per axis
    name = self._unique(prim.GetName())

    if kind == "DomeLight":
      texture = _light_input(prim, "texture:file", None)
      local = self.materials.resolve(texture) if texture is not None and getattr(texture, "path", "") else None
      display = colour * radiance * self.pixel_per_nit
      if local:
        print(
          f"  dome light {prim.GetPath()}: environment map {local}; render with "
          f"envmap={local} envmap_intensity={radiance * self.light_scale * colour.max():.6g}"
        )
      elif self.opt["dome"] == "skybox":
        if display.max() > 1.0:
          self.warn_once(f"{prim.GetPath()}: dome radiance {_fmt(display)} (display) clamped to 1 in the LDR skybox")
        self.skybox = np.minimum(display, 1.0) if self.skybox is None else np.minimum(self.skybox + display, 1.0)
      elif self.opt["dome"] == "ambient":
        self.ambient = display if self.ambient is None else self.ambient + display
      return
    if kind == "DistantLight":
      self.lights.append(_Light(name, "directional", pos, direction, colour, radiance, castshadow=shadow))
      return
    if kind == "SphereLight":
      r = float(_light_input(prim, "radius", 0.5)) * float(scale.mean())
      if bool(_light_input(prim, "treatAsPoint", False)):
        r = 0.0
      area = 4.0 * math.pi * r * r
      lum = radiance / area if normalize and area > 0 else radiance
      intensity = lum * math.pi * r * r if r > 0 else radiance
      self.lights.append(_Light(name, "spot" if cone else "point", pos, direction, colour, intensity, r, cone or 0.0, shadow))
      return
    if kind in ("RectLight", "DiskLight"):
      if kind == "RectLight":
        area = float(_light_input(prim, "width", 1.0)) * scale[0] * float(_light_input(prim, "height", 1.0)) * scale[1]
      else:
        area = math.pi * (float(_light_input(prim, "radius", 0.5)) * float(scale[:2].mean())) ** 2
      if _light_input(prim, "texture:file", None):
        self.warn_once(f"{prim.GetPath()}: light texture ignored")
      lum = radiance / area if normalize and area > 0 else radiance
      cutoff = cone if cone is not None else RECT_SPOT_CUTOFF
      self.lights.append(_Light(name, "spot", pos, direction, colour, lum * area, math.sqrt(area / math.pi), cutoff, shadow))
      return
    if kind == "CylinderLight":
      r = float(_light_input(prim, "radius", 0.5)) * float(scale[1:].mean())
      length = float(_light_input(prim, "length", 1.0)) * scale[0]
      area = 2.0 * math.pi * r * length
      lum = radiance / area if normalize and area > 0 else radiance
      self.lights.append(_Light(name, "point", pos, direction, colour, lum * 2.0 * r * length, r, 0.0, shadow))
      return
    print(f"  skipping light {prim.GetPath()}: {kind} is not supported")

  def warn_once(self, message: str):
    self.materials.warn(message)

  # Cameras ------------------------------------------------------------------------------------

  def camera(self, prim: Usd.Prim):
    cam = UsdGeom.Camera(prim)
    focal = float(cam.GetFocalLengthAttr().Get() or 50.0)
    aperture = float(cam.GetHorizontalApertureAttr().Get() or 20.955)
    fovy = math.degrees(2.0 * math.atan(0.5 * aperture / focal))
    self.cameras.append((self._unique(prim.GetName()), self.world(prim), fovy))

  # Traversal and output ----------------------------------------------------------------------

  def run(self) -> Path:
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    it = iter(Usd.PrimRange(self.stage.GetPseudoRoot(), predicate))
    for prim in it:
      imageable = UsdGeom.Imageable(prim)
      if imageable:
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
          it.PruneChildren()
          continue
        if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
          it.PruneChildren()
          continue
      if prim.IsA(UsdGeom.Mesh):
        self.mesh(prim)
      elif prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
        if prim.IsA(UsdGeom.PointBased):
          print(f"  skipping {prim.GetPath()}: {prim.GetTypeName()} is not supported")
        else:
          self.primitive(prim)
      elif prim.HasAPI(UsdLux.LightAPI):
        self.light(prim)
      elif prim.IsA(UsdGeom.Camera):
        self.camera(prim)
      elif prim.IsA(UsdGeom.PointInstancer):
        print(f"  skipping {prim.GetPath()}: PointInstancer is not supported")
    return self.write()

  def _bounds(self):
    points = [m[1] for m in self.meshes]
    for _, kind, local, affine, _ in self.prims:
      corners = np.array(np.meshgrid(*[[-s, s] for s in local])).reshape(3, -1).T
      points.append(corners @ affine[:3, :3].T + affine[:3, 3])
    if not points:
      return np.zeros(3), np.zeros(3)
    points = np.concatenate(points)
    return points.min(axis=0), points.max(axis=0)

  def write(self) -> Path:
    lower, upper = self._bounds()
    offset, factor = np.zeros(3), 1.0
    if self.opt["normalize"] is not None and self.meshes + self.prims:
      factor = self.opt["normalize"] / max(float((upper - lower).max()), 1e-12)
      offset = -np.array([0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), lower[2]])
    elif self.opt["ground"] and self.meshes + self.prims:
      offset = np.array([0.0, 0.0, -lower[2]])
    extent = float(np.linalg.norm(upper - lower)) * factor

    def place(p):
      return (np.asarray(p) + offset) * factor

    mesh_xml, geom_xml = [], []
    for name, pos, normal, uv, faces, mat in self.meshes:
      file = f"meshes/{name}.obj"
      _write_obj(self.out_dir / file, place(pos), normal, uv, faces)
      mesh_xml.append(f'    <mesh name="{name}" file="{file}" inertia="shell"/>')
      mat_attr = f' material="{mat}"' if mat else ""
      geom_xml.append(
        f'    <geom name="{name}" type="mesh" mesh="{name}"{mat_attr} contype="0" conaffinity="0" group="1" mass="0"/>'
      )
    for name, kind, local, affine, mat in self.prims:
      rot, scale = _orthonormalize(affine[:3, :3])
      size = local * scale * factor
      if kind == "sphere" and not np.allclose(size, size[0], rtol=1e-4):
        kind = "ellipsoid"
      if kind == "sphere":
        size = size[:1]
      elif kind in ("capsule", "cylinder"):
        size = np.array([0.5 * (size[0] + size[1]), size[2]])
      elif kind == "plane":
        size = np.array([size[0], size[1], 0.1 * max(size[0], size[1], 1e-3)])
      mat_attr = f' material="{mat}"' if mat else ""
      geom_xml.append(
        f'    <geom name="{name}" type="{kind}" size="{_fmt(size)}" pos="{_fmt(place(affine[:3, 3]))}" '
        f'quat="{_fmt(_mat_to_quat(rot))}"{mat_attr} contype="0" conaffinity="0" group="1" mass="0"/>'
      )

    light_xml = []
    light_range = max(10.0, 4.0 * extent)
    for li in self.lights:
      peak = float(li.colour.max())
      if peak <= 0.0 or li.intensity <= 0.0:
        continue
      intensity = li.intensity * self.light_scale * peak
      if li.kind != "directional":
        intensity *= factor * factor
      attrs = (
        f'name="{li.name}" type="{li.kind}" pos="{_fmt(place(li.pos))}" dir="{_fmt(li.dir)}" '
        f'diffuse="{_fmt(li.colour / peak)}" specular="0 0 0" intensity="{intensity:.6g}" '
        f'castshadow="{str(li.castshadow).lower()}"'
      )
      if li.kind != "directional":
        attrs += f' range="{light_range:.6g}" bulbradius="{li.bulbradius * factor:.6g}"'
      if li.kind == "spot":
        attrs += f' cutoff="{li.cutoff:.6g}"'
      light_xml.append(f"    <light {attrs}/>")
    for name, affine, fovy in self.cameras:
      x = affine[:3, 0] / max(np.linalg.norm(affine[:3, 0]), 1e-12)
      y = affine[:3, 1] / max(np.linalg.norm(affine[:3, 1]), 1e-12)
      light_xml.append(
        f'    <camera name="{name}" pos="{_fmt(place(affine[:3, 3]))}" xyaxes="{_fmt([*x, *y])}" fovy="{fovy:.6g}"/>'
      )

    visual = ""
    if self.lights or self.ambient is not None or self.skybox is not None:
      # The scene brings its own lighting: no headlight; a uniform dome becomes its ambient.
      ambient = self.ambient if self.ambient is not None else np.zeros(3)
      visual = f'  <visual>\n    <headlight ambient="{_fmt(ambient)}" diffuse="0 0 0" specular="0 0 0"/>\n  </visual>\n'
    sky = ""
    if self.skybox is not None:
      rgb = _linear_to_srgb(self.skybox)
      sky = f'    <texture name="dome" type="skybox" builtin="gradient" rgb1="{_fmt(rgb)}" rgb2="{_fmt(rgb)}" width="16" height="16"/>\n'
    textures = "\n".join(
      f'    <texture name="{n}" type="2d" file="{f}" colorspace="{c}"/>' for n, f, c in self.textures.entries
    )
    materials = "\n".join(xml for _, xml in self.mjcf_materials.values())
    xml = f"""<mujoco model="{self.name}">
  <!-- Converted from {self.path.name} by tools/usd_to_mjcf.py -->
  <compiler meshdir="." texturedir="."/>
{visual}  <asset>
{sky}{textures}
{materials}
{chr(10).join(mesh_xml)}
  </asset>
  <worldbody>
{chr(10).join(light_xml)}
{chr(10).join(geom_xml)}
  </worldbody>
</mujoco>
"""
    out = self.out_dir / f"{self.name}.xml"
    out.write_text(xml)
    print(
      f"  {len(self.meshes)} meshes, {len(self.prims)} primitives, {len(self.mjcf_materials)} materials, "
      f"{len(self.textures.entries)} textures, {len(self.lights)} lights, {len(self.cameras)} cameras "
      f"(light scale {self.light_scale:.4g}, {self.pixel_per_nit:.4g} per nit)"
    )
    return out


def convert(
  path: Path,
  out_dir: Path,
  name: str | None = None,
  scale: float = 1.0,
  normalize: float | None = None,
  ground: bool = False,
  url_roots: dict[str, str] | None = None,
  joint_positions: dict[str, float] | None = None,
  variants: dict[str, str] | None = None,
  dome: str = "skybox",
  ev100: float = DEFAULT_EV100,
  camera: str | None = None,
  preview_emission_scale: float = 1.0,
  max_texture_size: int = 2048,
) -> Path:
  """Convert a USD stage and return the path of the written MJCF file.

  Args:
    path: USD file.
    out_dir: output directory (MJCF, meshes/, textures/).
    name: model name (default: file stem).
    scale: extra uniform scale after metersPerUnit.
    normalize: if set, centre in x/y, rest on z = 0 and scale the largest extent to this size.
    ground: translate in z so that the model rests on z = 0 (no scaling).
    url_roots: http(s) URL prefix -> local directory, for absolute asset URLs.
    joint_positions: UsdPhysics joint name -> position (radians or metres) to pose before baking.
    variants: "[prim path:]variant set" -> selection (default prim when no path is given).
    dome: uniform DomeLight as "skybox" (clamped to the LDR range), "ambient" (unclamped) or
      "drop".
    ev100: exposure the light intensities are calibrated for (the renderer's default).
    camera: name of the camera whose exposure attributes calibrate the lights.
    preview_emission_scale: display-relative scale of UsdPreviewSurface emissiveColor.
    max_texture_size: textures are downscaled to at most this many texels per side.
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  options = {
    "scale": scale,
    "normalize": normalize,
    "ground": ground,
    "url_roots": url_roots or {},
    "joint_positions": joint_positions or {},
    "variants": variants or {},
    "dome": dome,
    "ev100": ev100,
    "camera": camera,
    "preview_emission_scale": preview_emission_scale,
    "max_texture_size": max_texture_size,
  }
  print(f"usd_to_mjcf {path}")
  return _Converter(Path(path), out_dir, name or Path(path).stem, options).run()


def _pairs(values: list[str]) -> dict[str, str]:
  return dict(v.split("=", 1) for v in values)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("usd", type=Path)
  parser.add_argument("out_dir", type=Path)
  parser.add_argument("--name")
  parser.add_argument("--scale", type=float, default=1.0)
  parser.add_argument("--normalize", type=float, help="target largest extent in metres")
  parser.add_argument("--ground", action="store_true", help="translate in z to rest on z = 0")
  parser.add_argument("--url-root", action="append", default=[], help="URL_PREFIX=LOCAL_DIR for absolute asset URLs")
  parser.add_argument("--joint", action="append", default=[], help="JOINT=VALUE (radians / metres)")
  parser.add_argument("--variant", action="append", default=[], help="[PRIM:]SET=VALUE")
  parser.add_argument("--dome", choices=["skybox", "ambient", "drop"], default="skybox")
  parser.add_argument("--ev100", type=float, default=DEFAULT_EV100)
  parser.add_argument("--camera", help="camera whose exposure calibrates the lights")
  parser.add_argument("--preview-emission-scale", type=float, default=1.0)
  parser.add_argument("--max-texture-size", type=int, default=2048)
  args = parser.parse_args()
  print(
    convert(
      args.usd,
      args.out_dir,
      args.name,
      args.scale,
      args.normalize,
      args.ground,
      _pairs(args.url_root),
      {k: float(v) for k, v in _pairs(args.joint).items()},
      _pairs(args.variant),
      args.dome,
      args.ev100,
      args.camera,
      args.preview_emission_scale,
      args.max_texture_size,
    )
  )


if __name__ == "__main__":
  main()
