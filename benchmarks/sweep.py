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

"""Benchmark sweeps: one bench.py process per configuration, results appended as JSONL.

Presets:
  score:   the fixed set behind the optimisation score: 3 scenes (one per group) × nworld
           {1024, 4096} × resolution {64, 128}, 1 spp, plus a 2×2-spp point. The score is the
           geometric mean of camera-frames/s.
  compare: mujoco_warp_pbr vs mujoco_warp (shadows on) vs Newton on the same scenes and sizes.

Usage:
  python benchmarks/sweep.py --preset score --out results/score.jsonl [--label NAME]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORE_SCENES = ["material_grid", "primitives", "damaged_helmet"]
ENVMAP = "studio_small_09_1k.hdr"


def _envmap_option() -> list[str]:
  asset_dir = Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "mujoco_warp_pbr"))
  path = asset_dir / "hdri" / ENVMAP
  return ["--set", f"envmap={json.dumps(str(path))}"] if path.exists() else []


def configurations(preset: str) -> list[list[str]]:
  configs = []
  if preset == "score":
    for scene in SCORE_SCENES:
      for nworld in (1024, 4096):
        for res in (64, 128):
          configs.append(["--scene", scene, "--nworld", str(nworld), "--res", str(res), *_envmap_option()])
    configs.append(
      ["--scene", "damaged_helmet", "--nworld", "1024", "--res", "64", "--set", "samples_per_pixel=2", *_envmap_option()]
    )
  elif preset == "compare":
    for scene in SCORE_SCENES + ["flight_helmet"]:
      for nworld, res in ((1024, 64), (4096, 64), (1024, 128)):
        size = ["--scene", scene, "--nworld", str(nworld), "--res", str(res)]
        configs.append([*size, "--renderer", "pbr", *_envmap_option()])
        configs.append([*size, "--renderer", "mjwarp", "--set", "use_shadows=true"])
        configs.append([*size, "--renderer", "newton"])
  else:
    raise ValueError(preset)
  return configs


def score(rows: list[dict]) -> float:
  """Geometric mean of camera-frames/s."""
  return math.exp(sum(math.log(r["frames_per_s"]) for r in rows) / len(rows))


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--preset", choices=["score", "compare"], default="score")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--label", default="", help="tag stored with every row (e.g. an experiment name)")
  parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="arguments appended to every bench.py call")
  args = parser.parse_args()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  for config in configurations(args.preset):
    cmd = [sys.executable, str(HERE / "bench.py"), *config, *args.extra]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0 or not lines:
      print("FAILED:", " ".join(config), result.stderr.strip().splitlines()[-1:] if result.stderr else "")
      continue
    row = json.loads(lines[-1])
    row["label"] = args.label
    rows.append(row)
    with args.out.open("a") as f:
      f.write(json.dumps(row) + "\n")
    print(
      f"{row['renderer']:7s} {row['scene']:15s} n={row['nworld']:5d} res={row['res']:4d} "
      f"{row['ms_per_frame_batch']:8.3f} ms  {row['frames_per_s']:12.0f} frames/s  {' '.join(row['options'])[:60]}"
    )
  if args.preset == "score" and rows:
    print(f"SCORE {args.label}: {score(rows):.0f} frames/s (geometric mean over {len(rows)} configurations)")


if __name__ == "__main__":
  main()
