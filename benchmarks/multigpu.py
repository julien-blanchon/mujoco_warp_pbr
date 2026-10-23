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

"""Multi-GPU scaling: the same bench.py configuration on 1..N GPUs at once, one process per GPU.

Batched rendering is data-parallel over worlds, so N GPUs each render their own worlds (as N
simulation processes would). Reports the aggregate frames/s and the scaling efficiency against
one GPU. Processes start timing together (MJPBR_BENCH_BARRIER), after each has compiled and
warmed up, so the measurements overlap.

  python benchmarks/multigpu.py --scene damaged_helmet --nworld 4096 --res 64 [--set KEY=VALUE]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gpu_count() -> int:
  out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
  return len(out.stdout.strip().splitlines())


def run(ngpu: int, bench_args: list[str]) -> list[dict]:
  procs = []
  barrier = tempfile.mkdtemp(prefix="mjpbr-barrier-")
  for gpu in range(ngpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MJPBR_BENCH_BARRIER=f"{barrier}:{ngpu}")
    cmd = [sys.executable, str(HERE / "bench.py"), *bench_args, "--allow-busy", "--repeats", "7"]
    procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
  rows = []
  for p in procs:
    out, _ = p.communicate()
    lines = [line for line in out.splitlines() if line.startswith("{")]
    rows.append(json.loads(lines[-1]) if lines else None)
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--out", type=Path)
  args, bench_args = parser.parse_known_args()
  total = gpu_count()
  single = None
  results = []
  for ngpu in sorted({1, *range(2, total + 1, 2), total}):
    rows = run(ngpu, bench_args)
    if any(r is None for r in rows):
      print(f"{ngpu} GPU(s): a process failed")
      continue
    fps = sum(r["frames_per_s"] for r in rows)
    single = single or fps
    entry = {"ngpu": ngpu, "frames_per_s": fps, "efficiency": fps / (single * ngpu), "device": rows[0]["device"]}
    results.append(entry)
    print(f"{ngpu} GPU(s): {fps:12.0f} frames/s  efficiency {entry['efficiency']:.2f}", flush=True)
  if args.out:
    args.out.write_text(json.dumps({"args": bench_args, "results": results}, indent=2))


if __name__ == "__main__":
  main()
