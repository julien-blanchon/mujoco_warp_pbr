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

"""Benchmark sweeps: one bench.py process per configuration, results appended as JSONL.

Presets:
  score:   the fixed set behind the optimisation score: 3 scenes (one per group) × nworld
           {1024, 4096} × resolution {64, 128}, 1 spp, plus a 2×2-spp point, all with identical
           cameras in every world. The score is the geometric mean of camera-frames/s. Add
           `--extra --jitter-camera` for per-world camera jitter (identical worlds overstate ray
           coherence); the rows in results/score.jsonl were measured without it (their
           "jitter_camera" field is false or absent).
  tune:    kernel options per GPU (register cap, block size, ordered traversal) on the score
           scenes at 4096 worlds, 64², with camera jitter.
  scaling: frames/s against batch size (1 to 32768 worlds at 64², 1 to 8192 at 128²) for
           mujoco_warp_pbr and mujoco_warp (shadows on).

Comparisons between renderers are benchmarks/suite.py's job.

Usage:
  python benchmarks/sweep.py --preset score --out results/score.jsonl [--label NAME] [--extra ...]
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import _common

HERE = Path(__file__).resolve().parent
# On a shared GPU, wait for other users' work to finish instead of benchmarking under contention.
BUSY_RETRIES = 60
BUSY_WAIT_S = 30
SCORE_SCENES = ["material_grid", "primitives", "damaged_helmet"]
PRESETS = ["score", "tune", "scaling"]


def _envmap_option() -> list[str]:
  """--set envmap=hdri/<name> (asset-relative, resolved by bench.py) if the HDRI is downloaded."""
  return ["--set", _common.ENVMAP_OPTION] if _common.envmap_available() else []


def configurations(preset: str) -> list[list[str]]:
  """bench.py argument lists of a preset."""
  configs = []
  if preset == "score":
    for scene in SCORE_SCENES:
      for nworld in (1024, 4096):
        for res in (64, 128):
          configs.append(["--scene", scene, "--nworld", str(nworld), "--res", str(res), *_envmap_option()])
    configs.append(
      ["--scene", "damaged_helmet", "--nworld", "1024", "--res", "64", "--set", "samples_per_pixel=2", *_envmap_option()]
    )
  elif preset == "tune":
    for scene in SCORE_SCENES:
      for regs in (0, 96, 128):
        for block in (64, 256):
          for ordered in ("true", "false"):
            options = ["--set", f"max_registers={regs}", "--set", f"block_dim={block}", "--set", f"ordered_traversal={ordered}"]
            configs.append(
              ["--scene", scene, "--nworld", "4096", "--res", "64", "--jitter-camera", *options, *_envmap_option()]
            )
  elif preset == "scaling":
    sizes = [(n, 64) for n in (1, 4, 16, 64, 256, 1024, 4096, 16384, 32768)]
    sizes += [(n, 128) for n in (1, 16, 256, 1024, 4096, 8192)]
    for scene in SCORE_SCENES:
      for nworld, res in sizes:
        size = ["--scene", scene, "--nworld", str(nworld), "--res", str(res)]
        configs.append([*size, "--renderer", "pbr", *_envmap_option()])
        configs.append([*size, "--renderer", "mjwarp", "--set", "use_shadows=true"])
  else:
    raise ValueError(preset)
  return configs


def run_bench(config: list[str]) -> dict | None:
  """Runs one bench.py configuration in a fresh process (waiting while other users occupy the
  GPU) and returns its JSON row, or None on failure."""
  cmd = [sys.executable, str(HERE / "bench.py"), *config]
  for _ in range(BUSY_RETRIES):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if "GPU in use" not in result.stderr:
      break
    time.sleep(BUSY_WAIT_S)
  lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
  if result.returncode != 0 or not lines:
    print("FAILED:", " ".join(config), result.stderr.strip().splitlines()[-1:] if result.stderr else "", flush=True)
    return None
  return json.loads(lines[-1])


def score(rows: list[dict]) -> float:
  """Geometric mean of camera-frames/s."""
  return math.exp(sum(math.log(r["frames_per_s"]) for r in rows) / len(rows))


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--preset", choices=PRESETS, default="score")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--label", default="", help="tag stored with every row (e.g. an experiment name)")
  parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="arguments appended to every bench.py call")
  args = parser.parse_args()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  for config in configurations(args.preset):
    row = run_bench([*config, *args.extra])
    if row is None:
      continue
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
