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

"""Pure-Python tests of the benchmark scripts (no GPU, no rendering)."""

import json
import os
from pathlib import Path

import _common
import bench
import cpu_bench
import pytest
import report
import suite
import sweep


def test_parse_options():
  options = _common.parse_options(["a=1", "b=true", "c=hdri/x.hdr", 'd="q"', "e=[1, 2]", "f=1.5", "g=a=b"])
  assert options == {"a": 1, "b": True, "c": "hdri/x.hdr", "d": "q", "e": [1, 2], "f": 1.5, "g": "a=b"}
  with pytest.raises(ValueError, match="KEY=VALUE"):
    _common.parse_options(["novalue"])


def test_resolve_options(tmp_path, monkeypatch):
  monkeypatch.setenv("MJPBR_ASSET_DIR", str(tmp_path))
  monkeypatch.chdir(tmp_path.parent)
  resolved = _common.resolve_options(_common.parse_options([_common.ENVMAP_OPTION, "tier=1"]))
  assert resolved == {"envmap": str(tmp_path / "hdri" / _common.ENVMAP), "tier": 1}
  assert _common.resolve_options({"envmap": "/abs/x.hdr"}) == {"envmap": "/abs/x.hdr"}
  assert _common.resolve_options({"envmap": ["hdri/a.hdr"]}) == {"envmap": [str(tmp_path / "hdri" / "a.hdr")]}
  assert _common.resolve_options({"tier": 1}) == {"tier": 1}


def test_file_barrier(tmp_path):
  _common.file_barrier(tmp_path, 1)  # this process alone
  with pytest.raises(TimeoutError):
    _common.file_barrier(tmp_path, 2, timeout=0.05)
  with _common.barrier_dir() as directory:
    assert os.path.isdir(directory)
  assert not os.path.exists(directory)


def test_time_steps():
  calls = []
  times = _common.time_steps(lambda: calls.append(1), iters=3, repeats=2)
  assert len(times) == 2 and len(calls) == 1 + 3 * 2
  calls.clear()
  _common.time_steps(lambda: calls.append(1), iters=3, repeats=2, warmup=False)
  assert len(calls) == 6


def test_sweep_configurations(tmp_path, monkeypatch):
  monkeypatch.setenv("MJPBR_ASSET_DIR", str(tmp_path))
  assert len(sweep.configurations("score")) == 13
  assert all("envmap" not in " ".join(c) for c in sweep.configurations("score"))  # HDRI not downloaded
  (tmp_path / "hdri").mkdir()
  (tmp_path / "hdri" / _common.ENVMAP).touch()
  score = sweep.configurations("score")
  assert all(_common.ENVMAP_OPTION in c for c in score)
  assert sum("samples_per_pixel=2" in c for c in score) == 1
  assert len(sweep.configurations("tune")) == 3 * 3 * 2 * 2
  scaling = sweep.configurations("scaling")
  assert len(scaling) == 3 * 15 * 2
  assert all(_common.ENVMAP_OPTION in c for c in scaling if "pbr" in c)
  with pytest.raises(ValueError):
    sweep.configurations("compare")
  assert sweep.score([{"frames_per_s": 10.0}, {"frames_per_s": 1000.0}]) == pytest.approx(100.0)


def test_suite_configurations():
  configs = suite.configurations(suite.SCENES, suite.CONTEXTS, suite.RENDERERS)
  keys = [json.dumps(k, sort_keys=True) for k, _ in configs]
  assert len(keys) == len(set(keys))
  for key, config in configs:
    lit = _common.ENVMAP_OPTION in config
    assert lit == (key["renderer"] in _common.ENV_RENDERERS and key["scene"] not in suite.scenes.LIGHTS_ONLY)
    if key["renderer"] in _common.SEQUENTIAL:
      assert key["context"] != "batch_64"
    assert config[config.index("--nworld") + 1] == str(key["nworld"])
  slow = [k for k, _ in configs if k["scene"] in suite.SLOW_NEWTON_IMPORT and k["renderer"] in ("newton", "newton_pbr")]
  assert slow and all(k["context"] == "batch_64" for k in slow)
  assert suite.set_options(["--renderer", "pbr", "--set", "tier=1", "--set", "denoise=true"]) == ["tier=1", "denoise=true"]


def _entry(scene: str, context: str, renderer: str, fps: float | None) -> dict:
  key = {"scene": scene, "context": context, "renderer": renderer, "nworld": 1, "res": 256}
  row = None if fps is None else {"frames_per_s": fps, "options": []}
  return {"key": key, "ok": fps is not None, "row": row, "wall_s": 1.0}


def test_report_render(tmp_path):
  run = tmp_path / "gpu"
  run.mkdir()
  system = {"gpus": ["Test GPU, 1.0, 1 MiB, 9.0"], "platform": "Linux", "packages": {"warp-lang": "1.0"}, "commit": "a" * 40}
  (run / "system.json").write_text(json.dumps(system))
  entries = [
    _entry("primitives", "single", "pbr", 2500.0),
    _entry("primitives", "single", "mjwarp", 4000.0),
    _entry("primitives", "single", "mitsuba", None),
    _entry("material_grid", "single", "pbr", 1.5e6),
    _entry("material_grid", "batch_64", "pbr", 900.0),
  ]
  (run / "results.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
  text = report.render([run])
  assert "## Test GPU" in text and "Commit `aaaaaaaaaaaa`." in text
  single = text[text.index("### Single scene") : text.index("### Batch: 4096")]
  assert "| scene | PBR real-time | MuJoCo Warp | Mitsuba (16 spp) |" in single
  assert "| primitives | 2.5k | **4.0k** | n/a |" in single
  assert "| material_grid | **1.50M** |  |  |" in single
  assert single.index("primitives |") < single.index("material_grid |")  # first-measured order
  assert "Across machines" not in text
  two = report.render([run, run])
  assert "## Across machines (PBR real-time)" in two


def test_report_renders_committed_suite():
  """Committed suite results (including rows of scenes no longer in the suite) still render."""
  root = Path(__file__).resolve().parents[1] / "results" / "suite" / "suite-v2"
  runs = sorted(p for p in root.iterdir() if (p / "results.jsonl").exists())
  if not runs:
    pytest.skip("no committed suite results")
  text = report.render(runs, base=root)
  assert text.startswith("# Benchmark suite")


def test_cpu_bench_cores(monkeypatch):
  assert cpu_bench._parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
  assert cpu_bench._parse_cpulist("") == []
  # Two NUMA domains of 4 CPUs, CPUs 4..7 / 12..15 being SMT siblings.
  monkeypatch.setattr(cpu_bench, "numa_domains", lambda allowed: [[c for c in (0, 1, 2, 3, 4, 5, 6, 7) if c in allowed]])
  monkeypatch.setattr(cpu_bench, "smt_rank", lambda cpu: int(cpu >= 4))
  assert cpu_bench.assign_cores(2, 2, list(range(8))) == [[0, 1], [2, 3]]
  assert cpu_bench.assign_cores(3, 2, list(range(8)))[2] == [4, 5]  # siblings once every core is used
  with pytest.raises(SystemExit):
    cpu_bench.assign_cores(5, 2, list(range(8)))


def test_cpu_bench_options(tmp_path, monkeypatch):
  monkeypatch.setenv("MJPBR_ASSET_DIR", str(tmp_path))
  opts = cpu_bench.options("pbr_path", "damaged_helmet", ["shadow_samples=4"], envmap=True)
  assert opts == {"tier": 1, "path_samples": 16, "denoise": True, "shadow_samples": 4, "envmap": f"hdri/{_common.ENVMAP}"}
  assert "envmap" not in cpu_bench.options("mjwarp", "damaged_helmet", [], envmap=True)
  assert "envmap" not in cpu_bench.options("pbr", "franka_showroom", [], envmap=True)  # lights only


def test_bench_args():
  args = bench.parse_args(["--scene", "primitives", "--set", "tier=1", "--set", _common.ENVMAP_OPTION])
  assert args.options["tier"] == 1 and args.options["envmap"].endswith(f"hdri/{_common.ENVMAP}")
  for bad in (
    ["--renderer", "mujoco", "--randomize"],
    ["--renderer", "newton", "--jitter-camera"],
    ["--renderer", "mujoco", "--set", "tier=1"],
    ["--renderer", "mitsuba", "--set", "tier=1"],
    ["--set", "novalue"],
    ["--gpus", "0", "1", "--save-image", "x.png"],
  ):
    with pytest.raises(SystemExit):
      bench.parse_args(bad)
  assert set(bench.RENDERERS) >= {"pbr", "mjwarp", "newton", "newton_pbr", "mujoco", "mitsuba"}


def test_bench_strip():
  argv = ["--gpus", "0", "1", "--scene", "x", "--out", "o.jsonl", "--repeats=3", "--set", "a=1"]
  out = bench._strip(bench._strip(bench._strip(argv, "--gpus", None), "--out", 1), "--repeats", 1)
  assert out == ["--scene", "x", "--set", "a=1"]
