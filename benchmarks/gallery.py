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

"""Side-by-side stills and orbit videos of the validation scenes:

  MuJoCo Warp (stock renderer) | real-time tier | path-traced tier | Mitsuba 3 reference

MuJoCo Warp has no environment lighting: in scenes lit only by an HDR environment its panel
uses MuJoCo's default headlight instead, so it is not black. Mitsuba path-traces the same
geometry, materials, lights and environment with the renderer's BRDF (filament_bsdf.py).

  python benchmarks/gallery.py --scene damaged_helmet grid_env --frames 48 --out results/gallery
"""

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import mitsuba as mi
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from references import filament_bsdf  # noqa: E402
from references import mitsuba_scene  # noqa: E402
from validate import VALIDATION  # noqa: E402
from validate import tone_map  # noqa: E402

import mujoco_warp_pbr as mjpbr  # noqa: E402
from mujoco_warp_pbr._src import ibl  # noqa: E402

_LABELS = ("MuJoCo Warp", "PBR real-time", "PBR path-traced", "Mitsuba 3 (reference)")


def _orbit(mjm: mujoco.MjModel, cam: int, pos0: np.ndarray, quat0: np.ndarray, angle: float):
  """Rotates a world-attached camera about the world z axis."""
  qz = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, qz)
  mjm.cam_pos[cam] = rot.reshape(3, 3) @ pos0
  quat = np.zeros(4)
  mujoco.mju_mulQuat(quat, qz, quat0)
  mjm.cam_quat[cam] = quat


class _Renderers:
  def __init__(self, mjm: mujoco.MjModel, env, res: int, path_samples: int):
    self.mjm = mjm
    self.env = env
    self.res = res
    self.cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    self.mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, self.mjd)
    kwargs = dict(cam_res=(res, res), cam_active=[self.cam], render_rgb=True, render_hdr=True)
    env_kwargs = dict(envmap=env, envmap_intensity=1.0) if env is not None else {}
    # Validation scenes are authored in display-relative radiance: expose at 1.
    pbr_kwargs = dict(kwargs, ev100=float(np.log2(1.0 / 1.2)), **env_kwargs)
    self.realtime = mjpbr.create_render_context(mjm, samples_per_pixel=2, **pbr_kwargs)
    self.path = mjpbr.create_render_context(mjm, tier=mjpbr.Tier.PATH, path_samples=path_samples, max_depth=4, **pbr_kwargs)
    self.exposure = float(self.realtime.exposure.numpy()[0])

    # Stock MuJoCo Warp, with the default headlight if nothing else lights the scene.
    self.mjm_stock = copy.deepcopy(mjm)
    if mjm.nlight == 0:
      self.mjm_stock.vis.headlight.active = 1
    self.stock = mjw.create_render_context(self.mjm_stock, **{k: v for k, v in kwargs.items() if k != "render_hdr"})

  def frame(self, angle: float, pos0, quat0, spp: int) -> list[np.ndarray]:
    for mjm in (self.mjm, self.mjm_stock):
      _orbit(mjm, self.cam, pos0, quat0, angle)
    mujoco.mj_forward(self.mjm, self.mjd)
    images = []

    m = mjw.put_model(self.mjm_stock)
    d = mjw.put_data(self.mjm_stock, self.mjd)
    mjw.refit_bvh(m, d, self.stock)
    mjw.render(m, d, self.stock)
    out = wp.zeros((1, self.res, self.res), dtype=wp.vec3)
    mjw.get_rgb(self.stock, 0, out)
    images.append(out.numpy()[0])

    m = mjw.put_model(self.mjm)
    d = mjw.put_data(self.mjm, self.mjd)
    for rc in (self.realtime, self.path):
      mjw.refit_bvh(m, d, rc.base)
      mjpbr.render(m, d, rc)
      hdr = wp.zeros((1, self.res, self.res), dtype=wp.vec3)
      mjpbr.get_hdr(rc, 0, hdr)
      images.append(tone_map(hdr.numpy()[0], self.exposure))

    scene = mitsuba_scene.scene_dict(
      self.mjm, self.mjd, self.realtime, self.cam, self.res, self.res, spp, envmap=self.env, integrator="path", max_depth=5
    )
    images.append(tone_map(mitsuba_scene.render(scene), self.exposure))
    return images


def _sheet(images: list[np.ndarray]) -> Image.Image:
  strip = np.concatenate([np.clip(i, 0.0, 1.0) for i in images], axis=1)
  sheet = Image.fromarray((strip * 255 + 0.5).astype(np.uint8))
  draw = ImageDraw.Draw(sheet)
  width = images[0].shape[1]
  for i, label in enumerate(_LABELS):
    draw.rectangle((i * width, 0, i * width + 8 + 7 * len(label), 16), fill=(0, 0, 0))
    draw.text((i * width + 4, 2), label, fill=(255, 255, 255))
  return sheet


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", nargs="*", default=["damaged_helmet", "water_bottle", "grid_env", "grid_sun"])
  parser.add_argument("--res", type=int, default=320)
  parser.add_argument("--frames", type=int, default=0, help="orbit video frames (0: still only)")
  parser.add_argument("--path-samples", type=int, default=256, help="samples per pixel of the path-traced tier")
  parser.add_argument("--spp", type=int, default=256, help="Mitsuba samples per pixel")
  parser.add_argument("--fps", type=int, default=24)
  parser.add_argument("--out", type=Path, default=Path("results/gallery"))
  args = parser.parse_args()

  mi.set_variant("cuda_ad_rgb")
  filament_bsdf.register(ibl.dfg_lut_array().numpy())
  args.out.mkdir(parents=True, exist_ok=True)
  for name in args.scene:
    make_model, make_env = VALIDATION[name]
    mjm = make_model()
    renderers = _Renderers(mjm, make_env() if make_env else None, args.res, args.path_samples)
    pos0, quat0 = mjm.cam_pos[renderers.cam].copy(), mjm.cam_quat[renderers.cam].copy()
    _sheet(renderers.frame(0.0, pos0, quat0, args.spp)).save(args.out / f"{name}.png")
    print(args.out / f"{name}.png")
    if args.frames <= 0:
      continue
    frame_dir = args.out / f"{name}_frames"
    frame_dir.mkdir(exist_ok=True)
    for f in range(args.frames):
      angle = 2.0 * np.pi * f / args.frames
      _sheet(renderers.frame(angle, pos0, quat0, args.spp)).save(frame_dir / f"{f:04d}.png")
    video = args.out / f"{name}.mp4"
    subprocess.run(
      ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(frame_dir / "%04d.png")]
      + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video)],
      check=True,
    )
    for frame in frame_dir.iterdir():
      frame.unlink()
    frame_dir.rmdir()
    print(video)


if __name__ == "__main__":
  main()
