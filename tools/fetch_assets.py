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

"""Download the benchmark assets into a local cache.

The cache root is $MJPBR_ASSET_DIR (default: ~/.cache/mujoco_warp_pbr). Assets are never
committed to the repository. Downloads are pinned (Khronos commit, Poly Haven md5) and
idempotent: files already present are skipped.

MuJoCo Menagerie robots are used as MJCF directly ($MJPBR_ASSET_DIR/menagerie/<robot>/).
Khronos models are also converted to MJCF (tools/gltf_to_mjcf.py, normalised to a 1 m extent)
into $MJPBR_ASSET_DIR/mjcf/<Model>/<Model>.xml.

NVIDIA Isaac Sim / IsaacLab USD scenes (--only isaac, needs usd-core from the `bench` extra) are
listed and downloaded from NVIDIA's public S3 bucket, pinned to one Isaac release and checked
against the listing's MD5 ETags, mirrored under $MJPBR_ASSET_DIR/isaac/, then converted with
tools/usd_to_mjcf.py into $MJPBR_ASSET_DIR/mjcf/<Scene>/<Scene>.xml. They are for local
benchmarking only: never commit or redistribute them (see ISAAC_LICENSE_NOTE). They are only
fetched when asked for explicitly.

Usage:
  python tools/fetch_assets.py [--only khronos|menagerie|hdri|isaac] [--list]
"""

import argparse
import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

USER_AGENT = "mujoco_warp_pbr-asset-fetch"

KHRONOS_COMMIT = "15cd02d304e54cb42a4db4eefe719f76190f4b68"
KHRONOS_REPO = "https://github.com/KhronosGroup/glTF-Sample-Assets.git"

# Model -> license. glTF-Binary is used when available, otherwise the multi-file glTF folder.
KHRONOS_MODELS = {
  # Material correctness (group 1).
  "MetalRoughSpheresNoTextures": "CC0-1.0",
  "MetalRoughSpheres": "CC-BY-4.0",
  "CompareMetallic": "CC0-1.0",
  "CompareRoughness": "CC0-1.0",
  "CompareBaseColor": "CC0-1.0",
  "CompareNormal": "CC0-1.0",
  "CompareAmbientOcclusion": "CC0-1.0",
  "NormalTangentTest": "CC0-1.0",
  "NormalTangentMirrorTest": "CC-BY-4.0",
  "TextureCoordinateTest": "CC0-1.0",
  "TextureEncodingTest": "CC0-1.0",
  "TextureSettingsTest": "CC-BY-4.0",
  "AlphaBlendModeTest": "CC-BY-4.0",
  "NegativeScaleTest": "CC-BY-4.0",
  "BoomBox": "CC0-1.0",
  # Asset-heavy PBR (group 3).
  "WaterBottle": "CC0-1.0",
  "DamagedHelmet": "CC-BY-4.0",
  "FlightHelmet": "CC0-1.0",
  "SciFiHelmet": "CC0-1.0",
  # PBR showcase scenes.
  "ABeautifulGame": "CC-BY-4.0",
  "Corset": "CC0-1.0",
  "Lantern": "CC0-1.0",
}

MENAGERIE_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Robot folder -> license (MJCF used as is).
MENAGERIE_MODELS = {
  "anybotics_anymal_c": "BSD-3-Clause",
  "franka_emika_panda": "Apache-2.0",
}

# Poly Haven HDRIs (all CC0), 1k equirectangular .hdr.
POLYHAVEN_HDRIS = [
  "studio_small_09",
  "brown_photostudio_02",
  "kloofendal_48d_partly_cloudy_puresky",
  "venice_sunset",
  "abandoned_workshop_02",
  "rural_asphalt_road",
]
HDRI_RESOLUTION = "1k"

# Largest extent (metres) of converted Khronos models.
MJCF_EXTENT = 1.0

# NVIDIA Isaac assets: public S3 bucket used as IsaacLab's asset root, pinned to one release.
ISAAC_S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ISAAC_RELEASE = "6.2"
ISAAC_PREFIX = f"Assets/Isaac/{ISAAC_RELEASE}/Isaac/"
ISAAC_MAX_MB = 150  # per scene, downloaded bytes
ISAAC_LICENSE_NOTE = (
  "NVIDIA Isaac assets: local benchmarking only, never commit or redistribute. Folder LICENSE files "
  "vary (FrankaEmika: Apache-2.0); Environments/environment-supplement-LICENSE.txt marks environments "
  "'Limited Use Content: for your use only, without modifications'."
)
# Folder under ISAAC_PREFIX -> (licence as stated by the folder, excluded key substrings).
ISAAC_FOLDERS = {
  "IsaacLab/Docs/Renderers/": ("no LICENSE file (IsaacLab docs asset)", ()),
  "IsaacLab/Robots/FrankaEmika/": ("Apache-2.0 (folder LICENSE)", ("/Legacy/", "_rotx_mimic_", ".orig", "/Props/")),
  "Props/Sektion_Cabinet/": ("no LICENSE file", ()),
  "Props/Mounts/SeattleLabTable/": ("no LICENSE file", ()),
}
# Franka Panda default pose of IsaacLab's FRANKA_PANDA_CFG (radians; fingers in metres).
FRANKA_POSE = {
  "panda_joint1": 0.0,
  "panda_joint2": -0.569,
  "panda_joint3": 0.0,
  "panda_joint4": -2.810,
  "panda_joint5": 0.0,
  "panda_joint6": 3.037,
  "panda_joint7": 0.741,
  "panda_finger_joint1": 0.04,
  "panda_finger_joint2": 0.04,
}
# Scene -> (folders, source USD relative to $MJPBR_ASSET_DIR/isaac or a composer, converter kwargs).
ISAAC_SCENES = {
  # IsaacLab renderer comparison scene: PreviewSurface spheres, textured floors, dome + rect light.
  "IsaacRendererGallery": (
    ["IsaacLab/Docs/Renderers/"],
    f"Isaac/{ISAAC_RELEASE}/Isaac/IsaacLab/Docs/Renderers/renderer_gallery_scene.usda",
    {},
  ),
  # Isaac-RenderBenchmark-Franka-Cabinet (IsaacLab render benchmark), one environment.
  "IsaacFrankaCabinet": (
    ["IsaacLab/Robots/FrankaEmika/", "Props/Sektion_Cabinet/"],
    "composed/franka_cabinet.usda",
    {"joint_positions": FRANKA_POSE},
  ),
  # IsaacLab's lift / stack table: OmniPBR with base colour, normal, metallic and roughness maps.
  "IsaacSeattleLabTable": (
    ["Props/Mounts/SeattleLabTable/"],
    f"Isaac/{ISAAC_RELEASE}/Isaac/Props/Mounts/SeattleLabTable/table.usd",
    {"ground": True},  # IsaacLab puts the table top at z = 0; the legs end at z = -1.04.
  ),
}


def asset_dir() -> Path:
  return Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "mujoco_warp_pbr"))


def _get(url: str) -> bytes:
  request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(request, timeout=120) as response:
    return response.read()


def _download(url: str, path: Path, md5: str | None = None):
  if path.exists():
    return
  data = _get(url)
  if md5 is not None and hashlib.md5(data).hexdigest() != md5:
    raise RuntimeError(f"md5 mismatch for {url}")
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".part")
  tmp.write_bytes(data)
  tmp.rename(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def fetch_khronos(root: Path):
  # Blobless sparse checkout: only the selected model folders are materialised.
  repo = root / "khronos" / "glTF-Sample-Assets"
  if not repo.exists():
    repo.mkdir(parents=True)
    _git(repo, "init", "--quiet")
    _git(repo, "remote", "add", "origin", KHRONOS_REPO)
  _git(repo, "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", KHRONOS_COMMIT)

  patterns = []
  for model, license_id in KHRONOS_MODELS.items():
    variants = _git(repo, "ls-tree", "--name-only", KHRONOS_COMMIT, f"Models/{model}/").split()
    variant = "glTF-Binary" if f"Models/{model}/glTF-Binary" in variants else "glTF"
    patterns += [f"/Models/{model}/{variant}/", f"/Models/{model}/LICENSE.md"]
    print(f"khronos {model:30s} {variant:12s} {license_id}")
  _git(repo, "sparse-checkout", "set", "--no-cone", *patterns)
  _git(repo, "checkout", "--quiet", KHRONOS_COMMIT)

  from gltf_to_mjcf import convert  # sibling module in tools/

  for model in KHRONOS_MODELS:
    out = root / "mjcf" / model
    if (out / f"{model}.xml").exists():
      continue
    src = sorted((repo / "Models" / model).glob("*/*.glb")) or sorted((repo / "Models" / model).glob("*/*.gltf"))
    convert(src[0], out, model, normalize=MJCF_EXTENT)


def fetch_menagerie(root: Path):
  repo = root / "menagerie"
  if not repo.exists():
    repo.mkdir(parents=True)
    _git(repo, "init", "--quiet")
    _git(repo, "remote", "add", "origin", MENAGERIE_REPO)
  _git(repo, "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT)
  _git(repo, "sparse-checkout", "set", "--no-cone", *[f"/{model}/" for model in MENAGERIE_MODELS])
  _git(repo, "checkout", "--quiet", MENAGERIE_COMMIT)
  for model, license_id in MENAGERIE_MODELS.items():
    print(f"menagerie {model:28s} {license_id}")


def fetch_hdris(root: Path):
  for hdri in POLYHAVEN_HDRIS:
    info = json.loads(_get(f"https://api.polyhaven.com/files/{hdri}"))["hdri"][HDRI_RESOLUTION]["hdr"]
    _download(info["url"], root / "hdri" / f"{hdri}_{HDRI_RESOLUTION}.hdr", info["md5"])
    print(f"hdri    {hdri:30s} {info['size'] / 1e6:.1f} MB CC0-1.0")


def _s3_list(prefix: str) -> list[tuple[str, int, str]]:
  """(key, size, etag) of every object under `prefix` (S3 ListObjectsV2, paginated)."""
  ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
  objects, token = [], None
  while True:
    query = {"list-type": "2", "prefix": prefix}
    if token:
      query["continuation-token"] = token
    tree = ET.fromstring(_get(f"{ISAAC_S3}/?{urllib.parse.urlencode(query)}"))
    for item in tree.findall("s3:Contents", ns):
      key = item.findtext("s3:Key", namespaces=ns)
      objects.append((key, int(item.findtext("s3:Size", namespaces=ns)), item.findtext("s3:ETag", namespaces=ns).strip('"')))
    if tree.findtext("s3:IsTruncated", namespaces=ns) != "true":
      return objects
    token = tree.findtext("s3:NextContinuationToken", namespaces=ns)


def _isaac_path(root: Path, key: str) -> Path:
  return root / "isaac" / key.removeprefix("Assets/")


def _s3_fetch(root: Path, objects: list[tuple[str, int, str]]):
  for key, _, etag in objects:
    # Single-part uploads have the MD5 as ETag; multipart ones ("-N" suffix) are not checked.
    _download(f"{ISAAC_S3}/{urllib.parse.quote(key)}", _isaac_path(root, key), None if "-" in etag else etag)


def _remote_refs(path: Path) -> set[str]:
  """S3 keys of absolute asset URLs (textures, MDLs) referenced from a downloaded USD layer."""
  from pxr import UsdUtils

  sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(path))
  host = ISAAC_S3 + "/"
  return {ref[len(host) :] for ref in [*sublayers, *references, *payloads] if ref.startswith(host)}


def _compose_franka_cabinet(root: Path) -> Path:
  """Isaac-RenderBenchmark-Franka-Cabinet's scene (one environment) as a small USD layer: robot,
  cabinet, ground cuboid, dome + distant light and the benchmark camera, as configured in
  IsaacLab's render_benchmark_env_cfg.py."""
  base = f"../Isaac/{ISAAC_RELEASE}/Isaac"
  eye, target = np.array([2.0, 0.0, 1.5]), np.array([0.4, 0.0, 0.4])
  forward = (target - eye) / np.linalg.norm(target - eye)
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  rot = np.stack([right, np.cross(right, forward), -forward], axis=1)  # USD camera looks along -Z.
  w = 0.5 * np.sqrt(max(1.0 + np.trace(rot), 1e-12))
  quat = [w, (rot[2, 1] - rot[1, 2]) / (4 * w), (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w)]
  text = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Scope "Looks"
    {{
        def Material "Ground"
        {{
            token outputs:surface.connect = </World/Looks/Ground/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.5, 0.5, 0.5)
                float inputs:metallic = 0
                token outputs:surface
            }}
        }}
    }}

    def Cube "Ground" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 1
        rel material:binding = </World/Looks/Ground>
        double3 xformOp:translate = (0, 0, -0.05)
        float3 xformOp:scale = (3, 3, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}

    def Xform "Robot" (
        prepend references = @{base}/IsaacLab/Robots/FrankaEmika/franka_panda.usda@
        variants = {{
            string Physics = "mujoco"
        }}
    )
    {{
        double3 xformOp:translate = (1, 0, 0)
        quatd xformOp:orient = (0, 0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}

    def Xform "Cabinet" (
        prepend references = @{base}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd@
    )
    {{
        double3 xformOp:translate = (0, 0, 0.4)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def DomeLight "Light"
    {{
        color3f inputs:color = (0.75, 0.75, 0.75)
        float inputs:intensity = 2000
    }}

    def DistantLight "LightDirectional"
    {{
        float inputs:angle = 0
        color3f inputs:color = (1, 1, 1)
        float inputs:intensity = 200
        bool inputs:normalize = 1
        quatd xformOp:orient = (0.8881, 0.3251, 0.3251, 0)
        uniform token[] xformOpOrder = ["xformOp:orient"]
    }}

    def Camera "cam"
    {{
        float focalLength = 24
        float horizontalAperture = 20.955
        float verticalAperture = 20.955
        float2 clippingRange = (0.05, 50)
        double3 xformOp:translate = ({eye[0]}, {eye[1]}, {eye[2]})
        quatd xformOp:orient = ({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
}}
"""
  path = root / "isaac" / "composed" / "franka_cabinet.usda"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)
  return path


def fetch_isaac(root: Path, scenes: list[str] | None = None):
  print(ISAAC_LICENSE_NOTE)
  listings = {}
  for scene, (folders, source, kwargs) in ISAAC_SCENES.items():
    if scenes and scene not in scenes:
      continue
    objects = []
    for folder in folders:
      if folder not in listings:
        _, exclude = ISAAC_FOLDERS[folder]
        listings[folder] = [
          o for o in _s3_list(ISAAC_PREFIX + folder) if "/.thumbs/" not in o[0] and not any(e in o[0] for e in exclude)
        ]
      objects += listings[folder]
    # Absolute URLs into the bucket (e.g. NVIDIA Base materials of another release) and, for
    # MDL modules, their texture folder (<module dir>/<module name>/).
    _s3_fetch(root, objects)
    extra, seen = [], set()
    layers = [_isaac_path(root, key) for key, _, _ in objects if key.endswith((".usd", ".usda", ".usdc"))]
    for key in sorted(set().union(set(), *[_remote_refs(p) for p in layers])):
      found = _s3_list(key)
      if key.endswith(".mdl"):
        found += _s3_list(key.removesuffix(".mdl") + "/")
      for o in found:
        if o[0] not in seen and "/.thumbs/" not in o[0]:
          seen.add(o[0])
          extra.append(o)
    total = sum(o[1] for o in objects + extra) / 1e6
    if total > ISAAC_MAX_MB:
      raise RuntimeError(f"{scene}: {total:.0f} MB exceeds the {ISAAC_MAX_MB} MB budget")
    _s3_fetch(root, extra)
    licences = ", ".join(f"{folder} {ISAAC_FOLDERS[folder][0]}" for folder in folders)
    print(f"isaac   {scene:30s} {total:6.1f} MB  {licences}" + ("; + remote NVIDIA Base materials" if extra else ""))

    from usd_to_mjcf import convert  # sibling module in tools/

    src = _compose_franka_cabinet(root) if source.startswith("composed/") else root / "isaac" / source
    out = root / "mjcf" / scene
    if not (out / f"{scene}.xml").exists():
      convert(src, out, scene, url_roots={f"{ISAAC_S3}/Assets/": str(root / "isaac")}, **kwargs)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--only", choices=["khronos", "menagerie", "hdri", "isaac"])
  parser.add_argument("--list", action="store_true", help="print the manifest and exit")
  parser.add_argument("--scene", action="append", choices=sorted(ISAAC_SCENES), help="Isaac scenes to fetch (default: all)")
  args = parser.parse_args()

  if args.list:
    for model, license_id in KHRONOS_MODELS.items():
      print(f"khronos {model:30s} {license_id}")
    for model, license_id in MENAGERIE_MODELS.items():
      print(f"menagerie {model:28s} {license_id}")
    for hdri in POLYHAVEN_HDRIS:
      print(f"hdri    {hdri:30s} CC0-1.0")
    for scene, (folders, _, _) in ISAAC_SCENES.items():
      print(f"isaac   {scene:30s} " + ", ".join(f"{f} {ISAAC_FOLDERS[f][0]}" for f in folders))
    print(ISAAC_LICENSE_NOTE)
    return

  root = asset_dir()
  print(f"asset dir: {root}")
  if args.only in (None, "khronos"):
    fetch_khronos(root)
  if args.only in (None, "menagerie"):
    fetch_menagerie(root)
  if args.only in (None, "hdri"):
    fetch_hdris(root)
  if args.only == "isaac":  # opt-in: licence-restricted, local use only
    fetch_isaac(root, args.scene)


if __name__ == "__main__":
  main()
