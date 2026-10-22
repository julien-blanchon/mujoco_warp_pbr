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

"""Benchmark suite: every renderer on every scene in three contexts, plus comparison videos.

Renderers:
  pbr          mujoco_warp_pbr, real-time tier
  pbr_path     mujoco_warp_pbr, path-traced tier (16 spp, denoised)
  newton_pbr   mujoco_warp_pbr through newton_pbr.SensorPbrCamera (scene imported into Newton)
  newton       Newton's SensorCamera (scene imported into Newton)
  newton_rtx   Newton's ViewerRTX: NVIDIA Omniverse RTX (RTX GPUs, newton[rtx]; one image at a time)
  mjwarp       MuJoCo Warp's renderer (shadows on)
  mujoco       MuJoCo's OpenGL renderer (one image at a time)
  mitsuba      Mitsuba 3 path tracer, 16 spp (one image at a time; reference quality at higher spp)

Contexts (one camera per world, identical cameras in every world):
  single       1 world at 256²: latency of rendering one scene
  batch_64     4096 worlds at 64²: RL-style batched rendering
  batch_128    1024 worlds at 128²

Sequential renderers (mujoco, mitsuba) are timed on a few images: their throughput does not
depend on the batch size. Writes <out>/system.json, <out>/results.jsonl (one bench.py row per
configuration, failures included), <out>/gallery/ (stills and orbit videos) and <out>/report.md.

  python benchmarks/suite.py --out results/suite/my-gpu [--quick] [--no-video]
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenes  # noqa: E402
from sweep import run_bench  # noqa: E402

SCENES = [
  "material_grid",
  "primitives",
  "damaged_helmet",
  "a_beautiful_game",
  "anymal_c",
  "franka_showroom",
  "isaac_franka_cabinet",
  "isaac_seattle_lab_table",
]
CONTEXTS = {"single": (1, 256), "batch_64": (4096, 64), "batch_128": (1024, 128)}
_SEQUENTIAL = ["--iters", "2", "--repeats", "3", "--sequential-frames", "8"]
RENDERERS = {
  "pbr": ["--renderer", "pbr"],
  "pbr_path": ["--renderer", "pbr", "--set", "tier=1", "--set", "path_samples=16", "--set", "denoise=true"],
  "newton_pbr": ["--renderer", "newton_pbr"],
  "newton": ["--renderer", "newton"],
  "mjwarp": ["--renderer", "mjwarp", "--set", "use_shadows=true"],
  "mujoco": ["--renderer", "mujoco", *_SEQUENTIAL],
  "newton_rtx": ["--renderer", "newton_rtx", *_SEQUENTIAL],
  "mitsuba": ["--renderer", "mitsuba", "--mitsuba-spp", "16", *_SEQUENTIAL],
}
# Renderers lit by the scene's HDR environment (the others only see the scene's own lights).
_ENV_RENDERERS = {"pbr", "pbr_path", "newton_pbr", "mitsuba"}
# Newton's MJCF import of these scenes takes minutes (mesh processing of >1M triangles): its two
# renderers are measured in batch_64 only.
SLOW_NEWTON_IMPORT = {"a_beautiful_game"}
GALLERY_PANELS = ["mujoco", "mjwarp", "newton", "newton_rtx", "pbr", "pbr_path", "mitsuba"]
ENVMAP = "studio_small_09_1k.hdr"


def _version(package: str) -> str | None:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return None


def system_info() -> dict:
  info = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "platform": platform.platform(),
    "cpu": platform.processor() or platform.machine(),
    "cpu_count": os.cpu_count(),
    "python": platform.python_version(),
    "packages": {p: _version(p) for p in ["warp-lang", "mujoco", "mujoco-warp", "newton", "mitsuba", "mujoco-warp-pbr"]},
  }
  try:
    query = "--query-gpu=name,driver_version,memory.total,compute_cap"
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True).stdout
    info["gpus"] = [line.strip() for line in out.strip().splitlines()]
  except OSError:
    info["gpus"] = []
  commit = HERE.parent / "COMMIT"
  if commit.exists():
    info["commit"] = commit.read_text().strip()
  else:
    try:
      info["commit"] = subprocess.run(
        ["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], capture_output=True, text=True
      ).stdout.strip()
    except OSError:
      pass
  return info


def configurations(scene_names: list[str], contexts: dict, renderers: dict) -> list[tuple[dict, list[str]]]:
  asset_dir = scenes.asset_dir()
  envmap = ["--set", f"envmap={json.dumps(str(asset_dir / 'hdri' / ENVMAP))}"]
  configs = []
  for scene in scene_names:
    for context, (nworld, res) in contexts.items():
      for renderer, args in renderers.items():
        sequential = renderer in ("mujoco", "mitsuba", "newton_rtx")
        if sequential and context == "batch_64":
          continue  # their per-image rate does not depend on nworld: 256² and 128² suffice
        if scene in SLOW_NEWTON_IMPORT:
          if renderer in ("newton", "newton_pbr") and context != "batch_64":
            continue
          if renderer == "newton_rtx" and context != "single":
            continue
        lit = envmap if renderer in _ENV_RENDERERS and scene not in scenes.LIGHTS_ONLY else []
        key = {"scene": scene, "context": context, "renderer": renderer, "nworld": nworld, "res": res}
        configs.append((key, ["--scene", scene, "--nworld", str(nworld), "--res", str(res), *args, *lit]))
  return configs


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--scene", nargs="*", default=SCENES, choices=sorted(scenes.SCENES))
  parser.add_argument("--renderer", nargs="*", default=list(RENDERERS), choices=list(RENDERERS))
  parser.add_argument("--context", nargs="*", default=list(CONTEXTS), choices=list(CONTEXTS))
  parser.add_argument("--quick", action="store_true", help="smaller batches and videos (smoke test)")
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument("--retry-failed", action="store_true", help="re-run configurations that failed before")
  parser.add_argument("--video-frames", type=int, default=48)
  parser.add_argument("--video-res", type=int, default=320)
  parser.add_argument("--video-spp", type=int, default=256, help="samples per pixel of the path-traced panels")
  args = parser.parse_args()
  os.environ.setdefault("MUJOCO_GL", "egl")

  contexts = {c: CONTEXTS[c] for c in args.context}
  if args.quick:
    contexts = {c: (min(n, 256), r) for c, (n, r) in contexts.items()}
    args.video_frames, args.video_res, args.video_spp = 8, 160, 32
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "system.json").write_text(json.dumps(system_info(), indent=2))

  results = args.out / "results.jsonl"
  done = set()
  if results.exists():  # resume: skip configurations already measured or failed
    kept = []
    for line in results.read_text().splitlines():
      entry = json.loads(line)
      if entry["ok"] or not args.retry_failed:
        done.add(json.dumps(entry["key"], sort_keys=True))
        kept.append(line)
    results.write_text("".join(line + "\n" for line in kept))
  configs = configurations(args.scene, contexts, {r: RENDERERS[r] for r in args.renderer})
  for i, (key, config) in enumerate(configs):
    if json.dumps(key, sort_keys=True) in done:
      continue
    start = time.time()
    row = run_bench(config)
    entry = {"key": key, "ok": row is not None, "row": row, "wall_s": round(time.time() - start, 1)}
    with results.open("a") as f:
      f.write(json.dumps(entry) + "\n")
    fps = f"{row['frames_per_s']:12.0f} frames/s" if row else "FAILED"
    print(f"[{i + 1}/{len(configs)}] {key['scene']:17s} {key['context']:9s} {key['renderer']:10s} {fps}", flush=True)

  if not args.no_video:
    import gallery

    for scene in args.scene:
      try:
        gallery.render_scene(
          scene,
          args.out / "gallery",
          args.video_res,
          args.video_frames,
          GALLERY_PANELS,
          path_samples=args.video_spp,
          spp=args.video_spp,
          fps=24,
        )
        print(f"gallery {scene}", flush=True)
      except Exception as e:  # keep the benchmark results even if one video fails
        print(f"gallery {scene} FAILED: {e}", flush=True)

  import report

  (args.out / "report.md").write_text(report.render([args.out]))
  print(args.out / "report.md")


if __name__ == "__main__":
  main()
