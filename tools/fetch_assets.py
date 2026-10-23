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

"""Download the benchmark assets into a local cache.

The cache root is $MJPBR_ASSET_DIR (default: ~/.cache/mujoco_warp_pbr). Assets are never
committed to the repository. Downloads are pinned (Khronos commit, Poly Haven md5) and
idempotent: files already present are skipped.

MuJoCo Menagerie robots are used as MJCF directly ($MJPBR_ASSET_DIR/menagerie/<robot>/).
Khronos models are also converted to MJCF (tools/gltf_to_mjcf.py, normalised to a 1 m extent)
into $MJPBR_ASSET_DIR/mjcf/<Model>/<Model>.xml.

Usage:
  python tools/fetch_assets.py [--only khronos|menagerie|hdri] [--list]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from _common import asset_dir

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


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--only", choices=["khronos", "menagerie", "hdri"])
  parser.add_argument("--list", action="store_true", help="print the manifest and exit")
  args = parser.parse_args()

  if args.list:
    for model, license_id in KHRONOS_MODELS.items():
      print(f"khronos {model:30s} {license_id}")
    for model, license_id in MENAGERIE_MODELS.items():
      print(f"menagerie {model:28s} {license_id}")
    for hdri in POLYHAVEN_HDRIS:
      print(f"hdri    {hdri:30s} CC0-1.0")
    return

  root = asset_dir()
  print(f"asset dir: {root}")
  if args.only in (None, "khronos"):
    fetch_khronos(root)
  if args.only in (None, "menagerie"):
    fetch_menagerie(root)
  if args.only in (None, "hdri"):
    fetch_hdris(root)


if __name__ == "__main__":
  main()
