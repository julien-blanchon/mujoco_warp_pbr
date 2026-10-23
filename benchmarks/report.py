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

"""Markdown report of one or more suite runs (one directory per machine).

python benchmarks/report.py results/suite/* > results/suite/README.md
"""

import argparse
import json
import os
from pathlib import Path

RENDERER_NAMES = {
  "pbr": "PBR real-time",
  "pbr_optix": "PBR real-time (OptiX)",
  "pbr_path": "PBR path (16 spp)",
  "newton_pbr": "PBR via Newton",
  "newton": "Newton",
  "newton_rtx": "Newton RTX",
  "newton_rtx_tiled": "Newton RTX (tiled)",
  "mjwarp": "MuJoCo Warp",
  "mujoco": "MuJoCo GL",
  "mitsuba": "Mitsuba (16 spp)",
}
CONTEXT_NAMES = {
  "single": "Single scene: 1 world, 256²",
  "batch_64": "Batch: 4096 worlds, 64²",
  "batch_128": "Batch: 1024 worlds, 128²",
}
SEQUENTIAL = {"mujoco", "mitsuba", "newton_rtx"}


def _fps(value: float) -> str:
  if value >= 1e6:
    return f"{value / 1e6:.2f}M"
  if value >= 1e3:
    return f"{value / 1e3:.1f}k"
  return f"{value:.0f}"


def load(run: Path) -> tuple[dict, dict]:
  system = json.loads((run / "system.json").read_text()) if (run / "system.json").exists() else {}
  results = {}
  path = run / "results.jsonl"
  if path.exists():
    for line in path.read_text().splitlines():
      entry = json.loads(line)
      k = entry["key"]
      results[(k["scene"], k["context"], k["renderer"])] = entry
  return system, results


def _gpu(system: dict, run: Path) -> str:
  gpus = system.get("gpus") or []
  return gpus[0].split(",")[0].strip() if gpus else run.name


def _table(results: dict, context: str) -> list[str]:
  scenes = sorted({k[0] for k in results if k[1] == context}, key=lambda s: list(dict.fromkeys(k[0] for k in results)).index(s))
  renderers = [r for r in RENDERER_NAMES if any(k[2] == r and k[1] == context for k in results)]
  if not scenes:
    return []
  lines = ["| scene | " + " | ".join(RENDERER_NAMES[r] for r in renderers) + " |"]
  lines.append("| --- |" + " ---: |" * len(renderers))
  for scene in scenes:
    cells = []
    for r in renderers:
      entry = results.get((scene, context, r))
      if entry is None:
        cells.append("")
      elif not entry["ok"]:
        cells.append("n/a")
      else:
        cell = _fps(entry["row"]["frames_per_s"])
        best = max(
          (e["row"]["frames_per_s"] for (s, c, _), e in results.items() if s == scene and c == context and e["ok"]),
          default=0,
        )
        cells.append(f"**{cell}**" if entry["row"]["frames_per_s"] == best else cell)
    lines.append(f"| {scene} | " + " | ".join(cells) + " |")
  return lines


def render(runs: list[Path], base: Path | None = None) -> str:
  """The report; image / video links are relative to `base` (default: the first run)."""
  base = base or runs[0]
  out = ["# Benchmark suite", ""]
  out += [
    "Camera-frames per second (one camera per world, identical cameras in every world; best per row in",
    "bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image",
    "throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene",
    "(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.",
    "",
  ]
  loaded = [(run, *load(run)) for run in runs]
  if len(loaded) > 1:
    out += ["## Across machines (PBR real-time)", ""]
    header = "| machine | " + " | ".join(CONTEXT_NAMES[c] for c in CONTEXT_NAMES) + " |"
    out += [header, "| --- |" + " ---: |" * len(CONTEXT_NAMES)]
    for run, system, results in loaded:
      cells = []
      for context in CONTEXT_NAMES:
        rows = [e["row"]["frames_per_s"] for (s, c, r), e in results.items() if c == context and r == "pbr" and e["ok"]]
        if rows:
          geo = 1.0
          for v in rows:
            geo *= v
          cells.append(_fps(geo ** (1.0 / len(rows))) + " (geo. mean)")
        else:
          cells.append("")
      out.append(f"| {_gpu(system, run)} | " + " | ".join(cells) + " |")
    out.append("")
  for run, system, results in loaded:
    out += [f"## {_gpu(system, run)}", ""]
    packages = ", ".join(f"{k} {v}" for k, v in system.get("packages", {}).items() if v)
    out += [f"{'; '.join(system.get('gpus', []))}. {system.get('platform', '')}. {packages}.", ""]
    if system.get("commit"):
      out += [f"Commit `{system['commit'][:12]}`.", ""]
    for context, title in CONTEXT_NAMES.items():
      table = _table(results, context)
      if table:
        out += [f"### {title}", "", *table, ""]
    gallery = run / "gallery"
    if gallery.exists():
      out += ["### Gallery", ""]
      out += ["MuJoCo GL | MuJoCo Warp | Newton | Newton RTX | PBR real-time | PBR path-traced | Mitsuba (reference).", ""]
      for still in sorted(gallery.glob("*.png")):
        rel = os.path.relpath(still, base)
        video = still.with_suffix(".mp4")
        link = f" ([orbit video]({os.path.relpath(video, base)}))" if video.exists() else ""
        out += [f"**{still.stem}**{link}", "", f"![{still.stem}]({rel})", ""]
  return "\n".join(out)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("runs", nargs="+", type=Path)
  parser.add_argument("--base", type=Path, help="directory the report is written to (for relative links)")
  args = parser.parse_args()
  print(render(args.runs, args.base))


if __name__ == "__main__":
  main()
