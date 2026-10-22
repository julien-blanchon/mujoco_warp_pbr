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

"""Denoiser quality: ꟻLIP of low-sample path-traced images, raw and denoised, against a
high-sample path-traced reference of the same scene.

  python benchmarks/denoise_eval.py --scene grid_env --samples 4 16 --set denoise_sigma_color=4.0
"""

import argparse
import json
from pathlib import Path

import flip_evaluator
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image
from validate import VALIDATION

import mujoco_warp_pbr as mjpbr


def _render(mjm, env, res: int, samples: int, denoise: bool, overrides: dict, seed: int = 0) -> np.ndarray:
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  kwargs = dict(envmap=env, envmap_intensity=1.0) if env is not None else {}
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
  parser.add_argument("--set", action="append", default=[], help="context attribute override KEY=VALUE")
  parser.add_argument("--out", type=Path, default=None, help="directory for comparison sheets")
  args = parser.parse_args()
  overrides = {k: float(v) if "." in v else int(v) for k, v in (s.split("=", 1) for s in args.set)}

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
        sheet = np.concatenate(images + [ref], axis=1)
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_{n}spp.png")


if __name__ == "__main__":
  main()
