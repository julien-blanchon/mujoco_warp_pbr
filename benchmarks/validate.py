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

"""Image quality of the real-time tier against Mitsuba 3.

Each validation scene is rendered by mujoco_warp_pbr (linear HDR) and by Mitsuba's `direct`
integrator (or `path`) with identical geometry, materials, lights and environment. Both images
go through the same exposure, tone mapper and sRGB encoding before ꟻLIP and PSNR; relMSE is
computed on linear HDR.

Expected differences: the real-time tier uses the split-sum approximation and ignores
environment visibility (no IBL shadows) and applies texture ambient occlusion, which a path
tracer replaces by true occlusion.

Usage:
  python benchmarks/validate.py [--scene NAME ...] [--res 256] [--spp 512]
    [--bsdf filament|principled] [--out results/validation]
"""

from __future__ import annotations

import argparse
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
from references import mitsuba_scene
from scenes import VALIDATION
from scenes import tone_map

import mujoco_warp_pbr as mjpbr

if TYPE_CHECKING:
  from _common import RGB
  from shape_extensions import Int
  from shape_extensions import IntVar


def render_ours[R: IntVar](
  mjm, env, res: Int[R], spp_axis: int, tier: int = mjpbr.Tier.REALTIME, path_samples: int = 256, max_depth: int = 4
) -> tuple[mujoco.MjData, mjpbr.PbrRenderContext, int, RGB[R, R]]:
  """Our linear HDR image (res, res, 3), with the data, render context and camera it came from."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  kwargs: dict[str, Any] = {"envmap": env, "envmap_intensity": 1.0} if env is not None else {}
  # Validation scenes are authored in display-relative radiance: expose at 1.
  rc = mjpbr.create_render_context(
    mjm,
    cam_res=(res, res),
    cam_active=[cam],
    render_rgb=True,
    render_hdr=True,
    samples_per_pixel=spp_axis,
    ev100=float(np.log2(1.0 / 1.2)),
    tier=tier,
    path_samples=path_samples,
    max_depth=max_depth,
    **kwargs,
  )
  mjw.refit_bvh(m, d, rc.base)
  mjpbr.render(m, d, rc)
  out = wp.zeros((1, res, res), dtype=wp.vec3)
  mjpbr.get_hdr(rc, 0, out)
  return mjd, rc, cam, out.numpy()[0]


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", nargs="*", default=sorted(VALIDATION))
  parser.add_argument("--res", type=int, default=256)
  parser.add_argument("--spp", type=int, default=512, help="Mitsuba samples per pixel")
  parser.add_argument("--aa", type=int, default=4, help="our sub-pixel samples per axis")
  parser.add_argument("--integrator", default="direct", choices=["direct", "path"])
  parser.add_argument("--bsdf", default="filament", choices=["filament", "principled"], help="Mitsuba-side material model")
  parser.add_argument("--tier", default="realtime", choices=["realtime", "path"], help="our rendering tier")
  parser.add_argument("--path-samples", type=int, default=512, help="our samples per pixel (path tier)")
  parser.add_argument("--max-depth", type=int, default=4, help="surface interactions per path (path tier)")
  parser.add_argument("--out", type=Path, default=Path("results/validation"))
  args = parser.parse_args()

  _common.setup_mitsuba("cuda_ad_rgb")
  args.out.mkdir(parents=True, exist_ok=True)
  rows = []
  for name in args.scene:
    make_model, make_env = VALIDATION[name]
    mjm = make_model()
    env = make_env() if make_env else None
    tier = mjpbr.Tier.PATH if args.tier == "path" else mjpbr.Tier.REALTIME
    mjd, rc, cam, ours = render_ours(mjm, env, args.res, args.aa, tier, args.path_samples, args.max_depth)
    # Mitsuba's max_depth counts the camera segment.
    scene = mitsuba_scene.scene_dict(
      mjm,
      mjd,
      rc,
      cam,
      args.res,
      args.res,
      args.spp,
      envmap=env,
      integrator=args.integrator,
      max_depth=args.max_depth + 1,
      bsdf=args.bsdf,
    )
    ref = mitsuba_scene.render(scene)

    exposure = float(rc.exposure.numpy()[0])
    a, b = tone_map(ours, exposure), tone_map(ref, exposure)
    flip_map, flip_mean, _ = flip_evaluator.evaluate(b.astype(np.float32), a.astype(np.float32), "LDR")
    mse = float(np.mean((a - b) ** 2))
    row = {
      "scene": name,
      "tier": args.tier,
      "integrator": args.integrator,
      "bsdf": args.bsdf,
      "res": args.res,
      "flip": float(flip_mean),
      "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
      "relmse": float(np.mean((ours - ref) ** 2 / (ref**2 + 1e-2))),
    }
    rows.append(row)
    print(json.dumps(row))
    # ours | Mitsuba | ꟻLIP error map, (res, 3 * res, 3).
    sheet = np.concatenate(
      [a, b, np.repeat(np.asarray(flip_map)[..., None], 3, -1) if np.asarray(flip_map).ndim == 2 else flip_map], axis=1
    )
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_{args.tier}_{args.bsdf}.png")
  with (args.out / f"metrics_{args.tier}_{args.bsdf}_{args.integrator}.jsonl").open("w") as f:
    for row in rows:
      f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
  main()
