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

"""Helpers shared by the benchmark scripts and tools.

Only the standard library is imported at module level, so lightweight scripts (sweep.py,
report.py, tools/hf/run_suite.py) can use it without the renderer's dependencies.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import numpy as np
  from shape_extensions import IntVar

# A float RGB image, (height, width, 3): display values in [0, 1] or linear radiance. Type-checking
# only (the alias is evaluated lazily), with the dtype left open (float32 or float64).
type RGB[H: IntVar, W: IntVar] = np.ndarray[[H, W, 3], np.dtype[Any]]

ROOT = Path(__file__).resolve().parents[1]

# The HDR environment of the benchmarks, relative to the asset directory.
ENVMAP = "studio_small_09_1k.hdr"
ENVMAP_OPTION = f"envmap=hdri/{ENVMAP}"
# suite.py / cpu_bench.py renderers lit by the scene's HDR environment (the others only see the
# scene's own lights).
ENV_RENDERERS = frozenset({"pbr", "pbr_optix", "pbr_path", "newton_pbr", "mitsuba"})
# bench.py renderers that draw one image per call: their throughput does not depend on nworld.
SEQUENTIAL = frozenset({"mujoco", "mitsuba", "newton_rtx"})


def asset_dir() -> Path:
  """The benchmark asset cache: $MJPBR_ASSET_DIR, default ~/.cache/mujoco_warp_pbr."""
  return Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "mujoco_warp_pbr"))


def parse_value(text: str) -> Any:
  """A --set value: JSON (numbers, true / false, quoted strings, lists) or else the raw string."""
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def parse_options(items: Iterable[str]) -> dict[str, Any]:
  """KEY=VALUE strings (--set) as a dict; raises ValueError for an item without `=`."""
  options: dict[str, Any] = {}
  for item in items:
    key, sep, value = item.partition("=")
    if not sep or not key:
      raise ValueError(f"expected KEY=VALUE, got {item!r}")
    options[key] = parse_value(value)
  return options


def resolve_asset(path: str) -> str:
  """A path as given if it is absolute or exists, otherwise relative to asset_dir()."""
  p = Path(path)
  return str(p if p.is_absolute() or p.exists() else asset_dir() / p)


def resolve_options(options: dict[str, Any]) -> dict[str, Any]:
  """Options with asset-relative `envmap` paths (e.g. "hdri/x.hdr") resolved against asset_dir()."""
  env = options.get("envmap")
  if isinstance(env, str):
    return {**options, "envmap": resolve_asset(env)}
  if isinstance(env, list) and all(isinstance(e, str) for e in env):
    return {**options, "envmap": [resolve_asset(e) for e in env]}
  return options


def envmap_available() -> bool:
  return (asset_dir() / "hdri" / ENVMAP).exists()


def run(cmd: list[str]) -> str | None:
  """stdout of a helper command (nvidia-smi, git, lscpu), or None if it is missing or fails."""
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  except OSError:
    return None
  return result.stdout if result.returncode == 0 else None


def nvidia_smi(query: str, device: str | None = None) -> list[list[str]] | None:
  """Rows of `nvidia-smi --query-...=<fields> --format=csv,noheader,nounits` (one per GPU or app).

  Args:
    query: e.g. "--query-gpu=utilization.gpu,memory.used".
    device: restrict to one GPU (index or PCI bus id, nvidia-smi's --id).
  """
  cmd = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
  if device is not None:
    cmd.append(f"--id={device}")
  out = run(cmd)
  if out is None:
    return None
  return [[field.strip() for field in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def package_version(package: str) -> str | None:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return None


def system_info(
  packages: Iterable[str] = ("warp-lang", "mujoco", "mujoco-warp", "newton", "mitsuba", "mujoco-warp-pbr"),
) -> dict:
  """Time, platform, package versions, GPUs and commit of a benchmark run."""
  info: dict[str, Any] = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "platform": platform.platform(),
    "cpu": platform.processor() or platform.machine(),
    "cpu_count": os.cpu_count(),
    "python": platform.python_version(),
    "packages": {p: package_version(p) for p in packages},
  }
  out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
  info["gpus"] = [line.strip() for line in out.strip().splitlines()] if out else []
  commit = ROOT / "COMMIT"  # written next to the code by tools/hf/run_suite.py
  if commit.exists():
    info["commit"] = commit.read_text().strip()
  elif (head := run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])) is not None:
    info["commit"] = head.strip()
  return info


def file_barrier(directory: str | Path, count: int, timeout: float = 900.0, poll: float = 0.02):
  """Waits until `count` processes have called file_barrier on the same directory.

  Raises:
    TimeoutError: fewer than `count` processes arrived within `timeout` seconds (one failed).
  """
  Path(directory, str(os.getpid())).touch()
  deadline = time.monotonic() + timeout
  while (arrived := len(os.listdir(directory))) < count:
    if time.monotonic() > deadline:
      raise TimeoutError(f"barrier {directory}: {arrived} of {count} processes after {timeout:.0f} s")
    time.sleep(poll)


@contextmanager
def barrier_dir(prefix: str = "mjpbr-barrier-") -> Generator[str]:
  """A temporary directory for file_barrier, removed afterwards."""
  directory = tempfile.mkdtemp(prefix=prefix)
  try:
    yield directory
  finally:
    shutil.rmtree(directory, ignore_errors=True)


def time_steps(
  step: Callable[[], Any], iters: int, repeats: int, sync: Callable[[], Any] = lambda: None, warmup: bool = True
) -> list[float]:
  """Seconds per step of `repeats` timed loops of `iters` steps (take the median).

  `sync` waits for the device (e.g. wp.synchronize) after the warm-up step and after each loop.
  """
  if warmup:
    step()
    sync()
  times = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(iters):
      step()
    sync()
    times.append((time.perf_counter() - start) / iters)
  return times


def setup_mitsuba(variant: str = "cuda_ad_rgb"):
  """Selects a Mitsuba variant and registers the renderer's BRDF (references/filament_bsdf.py).

  Returns:
    The mitsuba module.
  """
  import mitsuba as mi

  mi.set_variant(variant)
  from references import filament_bsdf

  from mujoco_warp_pbr._src import ibl

  filament_bsdf.register(ibl.dfg_lut_array().numpy())
  return mi
