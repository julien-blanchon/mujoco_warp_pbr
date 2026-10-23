# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.0"]
# ///
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

"""Run benchmarks/suite.py on several Hugging Face Jobs GPUs and collect the results.

A (private) Hugging Face bucket carries the code, the assets and the results:
  <bucket>/code      the repository at HEAD (git archive), synced by this script
  <bucket>/assets    $MJPBR_ASSET_DIR content (mjcf/, menagerie/, hdri/), synced with --sync-assets
  <bucket>/results/<run>/<flavor>   suite output of each job

Each job mounts the bucket at /data, runs tools/hf/bootstrap.sh and the suite, and writes its
results straight into the bucket. Jobs are billed per minute while they run; --timeout caps
each one. With --wait, the script polls until every job ends, downloads the results to
results/suite/<run>/ and writes a combined report there.

  uv run tools/hf/run_suite.py --bucket USER/BUCKET --flavors rtx-pro-6000 h200 a100-large \\
    --sync-assets --wait
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import Volume
from huggingface_hub import create_bucket
from huggingface_hub import inspect_job
from huggingface_hub import run_job

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
# Every job ends in one of these stages.
FINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def _sync(local: Path, remote: str, *extra: str):
  subprocess.run(["hf", "sync", str(local), remote, *extra], check=True)


def sync_code(bucket: str, prefix: str = "code"):
  """Uploads the repository at HEAD (tracked files) and its commit hash to <bucket>/<prefix>."""
  with tempfile.TemporaryDirectory() as tmp:
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    (Path(tmp) / "COMMIT").write_text(commit.stdout)
    _sync(Path(tmp), f"hf://buckets/{bucket}/{prefix}", "--delete")


def sync_assets(bucket: str, asset_dir: Path):
  for sub in ("mjcf", "menagerie", "hdri"):
    if (asset_dir / sub).exists():
      _sync(asset_dir / sub, f"hf://buckets/{bucket}/assets/{sub}", "--exclude", ".git/*")


def launch(bucket: str, flavor: str, run: str, image: str, timeout: str, suite_args: str) -> str:
  out = f"/data/results/{run}/{flavor}"
  script = (
    "set -eo pipefail; sh /data/code/tools/hf/bootstrap.sh; export PATH=$HOME/.local/bin:$PATH; cd /work/code; "
    "[ -d /work/optix-dev/include ] && export OPTIX_INCLUDE_DIR=/work/optix-dev/include; "
    f"mkdir -p {out}; uv run python benchmarks/suite.py --out {out} {suite_args} 2>&1 | tee {out}/log.txt"
  )
  job = run_job(
    image=image,
    command=["bash", "-c", script],
    flavor=flavor,
    timeout=timeout,
    env={"MJPBR_ASSET_DIR": "/data/assets", "MUJOCO_GL": "egl"},
    volumes=[Volume(type="bucket", source=bucket, mount_path="/data")],
    labels={"project": "mujoco-warp-pbr", "suite-run": run},
  )
  return job.id


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--bucket", required=True, help="Hugging Face bucket id, e.g. user/mujoco-warp-pbr")
  parser.add_argument("--flavors", nargs="+", default=["rtx-pro-6000", "h200", "a100-large"])
  parser.add_argument("--run", default=time.strftime("%Y%m%d-%H%M"), help="results sub-directory")
  parser.add_argument("--image", default=IMAGE)
  parser.add_argument("--timeout", default="3h", help="per-job cap, e.g. 90m or 3h")
  parser.add_argument("--suite-args", default="", help="extra benchmarks/suite.py arguments, e.g. --quick")
  parser.add_argument("--sync-assets", action="store_true", help="upload $MJPBR_ASSET_DIR to the bucket first")
  parser.add_argument("--asset-dir", type=Path, default=None)
  parser.add_argument("--no-sync-code", action="store_true", help="launch with the code already in the bucket")
  parser.add_argument("--wait", action="store_true", help="wait for the jobs, then download and report")
  args = parser.parse_args()

  create_bucket(args.bucket, private=True, exist_ok=True)
  if not args.no_sync_code:
    sync_code(args.bucket)
  if args.sync_assets:
    import os

    asset_dir = args.asset_dir or Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "mujoco_warp_pbr"))
    sync_assets(args.bucket, asset_dir)

  jobs = {}
  for flavor in args.flavors:
    jobs[flavor] = launch(args.bucket, flavor, args.run, args.image, args.timeout, args.suite_args)
    print(f"{flavor:16s} job {jobs[flavor]}  (hf jobs logs -f {jobs[flavor]})", flush=True)
  if not args.wait:
    return

  pending = dict(jobs)
  while pending:
    time.sleep(60)
    for flavor, job_id in list(pending.items()):
      stage = inspect_job(job_id=job_id).status.stage
      if stage in FINAL:
        print(f"{flavor:16s} {stage}", flush=True)
        del pending[flavor]
  local = ROOT / "results" / "suite" / args.run
  local.mkdir(parents=True, exist_ok=True)
  _sync_down = ["hf", "sync", f"hf://buckets/{args.bucket}/results/{args.run}", str(local)]
  subprocess.run(_sync_down, check=True)
  runs = sorted(p for p in local.iterdir() if (p / "results.jsonl").exists())
  report = subprocess.run(
    ["python", str(ROOT / "benchmarks" / "report.py"), *map(str, runs), "--base", str(local)],
    check=True,
    capture_output=True,
    text=True,
  ).stdout
  (local / "README.md").write_text(report)
  print(local / "README.md")


if __name__ == "__main__":
  main()
