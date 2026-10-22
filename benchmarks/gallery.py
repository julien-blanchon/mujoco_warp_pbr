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

"""Side-by-side stills and orbit videos of one scene through several renderers:

  mujoco      MuJoCo's OpenGL renderer
  mjwarp      MuJoCo Warp's ray-traced Lambert / Phong renderer
  newton      Newton's SensorCamera (the scene imported into Newton)
  newton_rtx  Newton's ViewerRTX: NVIDIA Omniverse RTX (RTX GPU, newton[rtx]; own "studio" lighting)
  pbr         mujoco_warp_pbr real-time tier
  pbr_path    mujoco_warp_pbr path-traced tier
  mitsuba     Mitsuba 3 path tracer with the renderer's BRDF (reference)

Scenes are benchmark scenes (`scenes.SCENES`, physical exposure, studio HDRI unless the scene is
lights-only) or validation scenes (`validate.VALIDATION`, display-relative exposure 1). Panels a
renderer cannot draw (e.g. Mitsuba without capsules, Newton for validation scenes) show "n/a".
MuJoCo and MuJoCo Warp have no environment lighting: in scenes lit only by an HDRI they use
MuJoCo's default headlight.

  python benchmarks/gallery.py --scene damaged_helmet franka_showroom --frames 60
"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenes  # noqa: E402
from validate import VALIDATION  # noqa: E402
from validate import tone_map  # noqa: E402

import mujoco_warp_pbr as mjpbr  # noqa: E402
from mujoco_warp_pbr._src import envmap as envmap_lib  # noqa: E402

LABELS = {
  "mujoco": "MuJoCo (OpenGL)",
  "mjwarp": "MuJoCo Warp",
  "newton": "Newton SensorCamera",
  "newton_rtx": "Newton RTX (ovrtx)",
  "pbr": "PBR real-time",
  "pbr_path": "PBR path-traced",
  "mitsuba": "Mitsuba 3 (reference)",
}
ENVMAP = "studio_small_09_1k.hdr"


def _orbit(mjm: mujoco.MjModel, cam: int, pos0: np.ndarray, quat0: np.ndarray, angle: float):
  """Rotates a world-attached camera about the world z axis."""
  qz = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, qz)
  mjm.cam_pos[cam] = rot.reshape(3, 3) @ pos0
  quat = np.zeros(4)
  mujoco.mju_mulQuat(quat, qz, quat0)
  mjm.cam_quat[cam] = quat


def load_scene(name: str):
  """(spec or None, model, environment map or None, display-relative exposure?) of a scene."""
  if name in scenes.SCENES:
    spec = scenes.SCENES[name][1]()
    env = None
    if name not in scenes.LIGHTS_ONLY:
      env = envmap_lib.load_hdr(scenes.asset_dir() / "hdri" / ENVMAP)
    return spec, spec.compile(), env, False
  make_model, make_env = VALIDATION[name]
  return None, make_model(), (make_env() if make_env else None), True


class Gallery:
  def __init__(self, name: str, res: int, panels: list[str], path_samples: int, spp: int):
    self.spec, self.mjm, self.env, unit_exposure = load_scene(name)
    self.res, self.panels, self.spp = res, panels, spp
    self.cam = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    self.pos0, self.quat0 = self.mjm.cam_pos[self.cam].copy(), self.mjm.cam_quat[self.cam].copy()
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    common = dict(cam_res=(res, res), cam_active=[self.cam], render_rgb=True)
    pbr = dict(common, render_hdr=True)
    if self.env is not None:
      pbr.update(envmap=self.env, envmap_intensity=1.0 if unit_exposure else None)
    if unit_exposure:
      pbr["ev100"] = float(np.log2(1.0 / 1.2))
    self.realtime = mjpbr.create_render_context(self.mjm, samples_per_pixel=2, **pbr)
    self.exposure = float(self.realtime.exposure.numpy()[0])
    self.path = None
    if "pbr_path" in panels:
      self.path = mjpbr.create_render_context(self.mjm, tier=mjpbr.Tier.PATH, path_samples=path_samples, **pbr)

    # MuJoCo-style renderers: the default headlight if nothing else lights the scene.
    self.mjm_stock = copy.deepcopy(self.mjm)
    if self.mjm.nlight == 0:
      self.mjm_stock.vis.headlight.active = 1
    self.stock = mjw.create_render_context(self.mjm_stock, **common) if "mjwarp" in panels else None
    self.gl = None
    if "mujoco" in panels:
      try:
        self.gl = mujoco.Renderer(self.mjm_stock, res, res)
      except Exception as e:  # no OpenGL context on this machine
        print(f"mujoco panel disabled: {e}")
    self.newton = None
    if "newton" in panels and self.spec is not None:
      from references import newton_camera

      self.newton = newton_camera.NewtonCamera(self.spec, self.mjm, 1, res)
    self.rtx = None
    if "newton_rtx" in panels and self.spec is not None:
      try:
        from references import newton_rtx

        self.rtx = newton_rtx.NewtonRtxCamera(self.spec, self.mjm, res)
      except Exception as e:  # needs an RTX GPU and newton[rtx]
        print(f"newton_rtx panel disabled: {e}")
    self.mitsuba = "mitsuba" in panels
    if self.mitsuba:
      import mitsuba as mi

      mi.set_variant("cuda_ad_rgb")
      from references import filament_bsdf

      from mujoco_warp_pbr._src import ibl

      filament_bsdf.register(ibl.dfg_lut_array().numpy())

  def frame(self, angle: float) -> list[np.ndarray | None]:
    for mjm in (self.mjm, self.mjm_stock):
      _orbit(mjm, self.cam, self.pos0, self.quat0, angle)
    mujoco.mj_forward(self.mjm, self.mjd)
    out = {}
    if self.gl is not None:
      self.gl.update_scene(self.mjd, camera=self.cam)
      out["mujoco"] = self.gl.render().astype(np.float32) / 255.0
    if self.stock is not None:
      m = mjw.put_model(self.mjm_stock)
      d = mjw.put_data(self.mjm_stock, self.mjd)
      mjw.refit_bvh(m, d, self.stock)
      mjw.render(m, d, self.stock)
      rgb = wp.zeros((1, self.res, self.res), dtype=wp.vec3)
      mjw.get_rgb(self.stock, 0, rgb)
      out["mjwarp"] = rgb.numpy()[0]
    if self.newton is not None:
      rot = self.mjd.cam_xmat[self.cam].reshape(3, 3).astype(np.float32)
      t = wp.transformf(wp.vec3f(*self.mjd.cam_xpos[self.cam]), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
      self.newton.cameras.assign(np.array([[t[i] for i in range(7)]], dtype=np.float32))
      self.newton.render()
      c = self.newton.color.numpy()[0]
      out["newton"] = np.stack([(c >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.float32) / 255.0
    if self.rtx is not None:
      _orbit(self.rtx.mjm, self.cam, self.pos0, self.quat0, angle)
      self.rtx.update_camera()
      self.rtx.render()
      out["newton_rtx"] = self.rtx.image()
    m = mjw.put_model(self.mjm)
    d = mjw.put_data(self.mjm, self.mjd)
    for name, rc in (("pbr", self.realtime), ("pbr_path", self.path)):
      if rc is None or name not in self.panels:
        continue
      mjw.refit_bvh(m, d, rc.base)
      mjpbr.render(m, d, rc)
      rgb = wp.zeros((1, self.res, self.res), dtype=wp.vec3)
      mjpbr.get_rgb(rc, 0, rgb)
      out[name] = rgb.numpy()[0]
    if self.mitsuba:
      from references import mitsuba_scene

      try:
        scene = mitsuba_scene.scene_dict(
          self.mjm,
          self.mjd,
          self.realtime,
          self.cam,
          self.res,
          self.res,
          self.spp,
          envmap=self.env,
          integrator="path",
          max_depth=5,
        )
        out["mitsuba"] = tone_map(mitsuba_scene.render(scene), self.exposure)
      except NotImplementedError as e:
        print(f"mitsuba panel disabled: {e}")
        self.mitsuba = False
    return [out.get(p) for p in self.panels]


def sheet(images: list[np.ndarray | None], panels: list[str], res: int) -> Image.Image:
  tiles = [np.clip(i, 0.0, 1.0) if i is not None else np.full((res, res, 3), 0.15) for i in images]
  img = Image.fromarray((np.concatenate(tiles, axis=1) * 255 + 0.5).astype(np.uint8))
  draw = ImageDraw.Draw(img)
  for i, (panel, image) in enumerate(zip(panels, images)):
    label = LABELS[panel] + ("" if image is not None else " (n/a)")
    draw.rectangle((i * res, 0, i * res + 8 + 7 * len(label), 16), fill=(0, 0, 0))
    draw.text((i * res + 4, 2), label, fill=(255, 255, 255))
  return img


def render_scene(name: str, out: Path, res: int, frames: int, panels: list[str], path_samples: int, spp: int, fps: int):
  """Writes <out>/<name>.png (first frame) and, with frames > 0, <out>/<name>.mp4 (orbit)."""
  gallery = Gallery(name, res, panels, path_samples, spp)
  out.mkdir(parents=True, exist_ok=True)
  sheet(gallery.frame(0.0), panels, res).save(out / f"{name}.png")
  if frames <= 0:
    return
  frame_dir = out / f"{name}_frames"
  frame_dir.mkdir(exist_ok=True)
  for f in range(frames):
    sheet(gallery.frame(2.0 * np.pi * f / frames), panels, res).save(frame_dir / f"{f:04d}.png")
  subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frame_dir / "%04d.png")]
    + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / f"{name}.mp4")],
    check=True,
  )
  for frame in frame_dir.iterdir():
    frame.unlink()
  frame_dir.rmdir()


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", nargs="*", default=["damaged_helmet", "water_bottle", "grid_env", "grid_sun"])
  parser.add_argument("--panels", nargs="*", default=["mjwarp", "pbr", "pbr_path", "mitsuba"], choices=sorted(LABELS))
  parser.add_argument("--res", type=int, default=320)
  parser.add_argument("--frames", type=int, default=0, help="orbit video frames (0: still only)")
  parser.add_argument("--path-samples", type=int, default=256, help="samples per pixel of the path-traced tier")
  parser.add_argument("--spp", type=int, default=256, help="Mitsuba samples per pixel")
  parser.add_argument("--fps", type=int, default=24)
  parser.add_argument("--out", type=Path, default=Path("results/gallery"))
  args = parser.parse_args()
  os.environ.setdefault("MUJOCO_GL", "egl")
  for name in args.scene:
    render_scene(name, args.out, args.res, args.frames, args.panels, args.path_samples, args.spp, args.fps)
    print(args.out / name)


if __name__ == "__main__":
  main()
