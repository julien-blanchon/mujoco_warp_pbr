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

"""Run the test suite on a Hugging Face Jobs GPU for the repository at HEAD.

The code goes to <bucket>/code-<tag> (so several checkouts can test at once), the job runs
tools/hf/bootstrap.sh and pytest, and the script waits and prints the log tail.

  uv run tools/hf/run_tests.py --bucket USER/BUCKET --tag mybranch -- -x -q tests/test_render.py
"""

import argparse
import sys
import time

from huggingface_hub import Volume
from huggingface_hub import fetch_job_logs
from huggingface_hub import inspect_job
from huggingface_hub import run_job
from run_suite import FINAL
from run_suite import IMAGE
from run_suite import sync_code


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--bucket", required=True, help="Hugging Face bucket id, e.g. user/mujoco-warp-pbr")
  parser.add_argument("--flavor", default="l40sx1")
  parser.add_argument("--tag", default="tests", help="code directory suffix in the bucket")
  parser.add_argument("--timeout", default="45m")
  parser.add_argument("--image", default=IMAGE)
  parser.add_argument("--tail", type=int, default=60, help="log lines printed at the end")
  parser.add_argument("pytest_args", nargs="*", default=["-q"], help="arguments after --")
  args = parser.parse_args()

  code = f"code-{args.tag}"
  sync_code(args.bucket, code)
  script = (
    f"set -eo pipefail; MJPBR_CODE=/data/{code} sh /data/{code}/tools/hf/bootstrap.sh; "
    "export PATH=$HOME/.local/bin:$PATH; cd /work/code; "
    "[ -d /work/optix-dev/include ] && export OPTIX_INCLUDE_DIR=/work/optix-dev/include; "
    f"uv run pytest {' '.join(args.pytest_args)}"
  )
  job = run_job(
    image=args.image,
    command=["bash", "-c", script],
    flavor=args.flavor,
    timeout=args.timeout,
    env={"MJPBR_ASSET_DIR": "/data/assets", "MUJOCO_GL": "egl"},
    volumes=[Volume(type="bucket", source=args.bucket, mount_path="/data")],
    labels={"project": "mujoco-warp-pbr", "tests": args.tag},
  )
  print(f"job {job.id}  (hf jobs logs -f {job.id})", flush=True)
  while (stage := inspect_job(job_id=job.id).status.stage) not in FINAL:
    time.sleep(30)
  print("\n".join(list(fetch_job_logs(job_id=job.id))[-args.tail :]))
  print(f"job {job.id}: {stage}")
  sys.exit(0 if stage == "COMPLETED" else 1)


if __name__ == "__main__":
  main()
