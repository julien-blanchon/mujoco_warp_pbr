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

"""Denoiser quality: ꟻLIP of low-sample path-traced images, raw and denoised, against a
high-sample path-traced reference of the same scene.

  python benchmarks/denoise_eval.py --scene grid_env --samples 4 16 --set denoise_sigma_color=4.0

--set KEY=VALUE applies to the denoised renders: a create_render_context keyword argument (e.g.
denoise_iterations=3) or an attribute of the render context (e.g. denoise_sigma_color=4.0);
unknown keys are an error. Values are parsed as in bench.py (JSON, else a string).
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _common
import flip_evaluator
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image
from scenes import VALIDATION

import mujoco_warp_pbr as mjpbr

if TYPE_CHECKING:
  from _common import RGB
  from shape_extensions import Int
  from shape_extensions import IntVar

_CONTEXT_KWARGS = set(inspect.signature(mjpbr.create_render_context).parameters)


def _render[R: IntVar](
  mjm, env, res: Int[R], samples: int, denoise: bool, overrides: dict[str, Any], seed: int = 0
) -> RGB[R, R]:
  """Display RGB (res, res, 3) of the path-traced tier."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  kwargs: dict[str, Any] = {"envmap": env, "envmap_intensity": 1.0} if env is not None else {}
  kwargs |= {k: v for k, v in overrides.items() if k in _CONTEXT_KWARGS}
  rc = mjpbr.create_render_context(
    mjm,
    cam_res=(res, res),
    cam_active=[cam],
    render_rgb=True,
    ev100=float(np.log2(1.0 / 1.2)),
    tier=mjpbr.Tier.PATH,
    path_samples=samples,
    denoise=denoise,
    **kwargs,
  )
  for key, value in overrides.items():
    if key not in _CONTEXT_KWARGS:
      if not hasattr(rc, key):
        raise SystemExit(f"--set {key}: neither a create_render_context argument nor a render context attribute")
      setattr(rc, key, value)
  mjw.refit_bvh(m, d, rc.base)
  mjpbr.render(m, d, rc, seed=seed)
  out = wp.zeros((1, res, res), dtype=wp.vec3)
  mjpbr.get_rgb(rc, 0, out)
  return out.numpy()[0]


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", nargs="*", default=["grid_env", "grid_sun", "damaged_helmet", "water_bottle"])
  parser.add_argument("--res", type=int, default=256)
  parser.add_argument("--samples", type=int, nargs="*", default=[1, 4, 16])
  parser.add_argument("--reference-samples", type=int, default=4096)
  parser.add_argument("--set", action="append", default=[], help="denoised-render option KEY=VALUE (see above)")
  parser.add_argument("--out", type=Path, default=None, help="directory for comparison sheets")
  args = parser.parse_args()
  try:
    overrides = _common.parse_options(args.set)
  except ValueError as e:
    parser.error(str(e))

  for name in args.scene:
    make_model, make_env = VALIDATION[name]
    mjm = make_model()
    env = make_env() if make_env else None
    ref = _render(mjm, env, args.res, args.reference_samples, False, {}, seed=12345)
    for n in args.samples:
      row = {"scene": name, "samples": n, "options": args.set}
      images = []
      for denoise in (False, True):
        img = _render(mjm, env, args.res, n, denoise, overrides if denoise else {})
        _, flip, _ = flip_evaluator.evaluate(ref.astype(np.float32), img.astype(np.float32), "LDR")
        row["flip_denoised" if denoise else "flip_raw"] = float(flip)
        images.append(img)
      print(json.dumps(row))
      if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        sheet = np.concatenate([*images, ref], axis=1)  # raw | denoised | reference, (res, 3 * res, 3)
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_{n}spp.png")


if __name__ == "__main__":
  main()
