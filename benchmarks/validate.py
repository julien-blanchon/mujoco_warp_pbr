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

import argparse
import json
import sys
from pathlib import Path

import flip_evaluator
import mitsuba as mi
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenes  # noqa: E402
from references import filament_bsdf  # noqa: E402
from references import mitsuba_scene  # noqa: E402

import mujoco_warp_pbr as mjpbr  # noqa: E402
from mujoco_warp_pbr._src import envmap as envmap_lib  # noqa: E402
from mujoco_warp_pbr._src import ibl  # noqa: E402


def _grid(light: str, floor: bool = False) -> mujoco.MjModel:
  n = 5
  mats, geoms = [], []
  for i in range(n):
    for j in range(n):
      mats.append(f'<material name="m{i}{j}" rgba="0.95 0.64 0.54 1" metallic="{i / (n - 1)}" roughness="{j / (n - 1)}"/>')
      geoms.append(f'<geom type="sphere" size="0.09" pos="{(j - 2) * 0.22:.2f} 0 {(i - 2) * 0.22:.2f}" material="m{i}{j}"/>')
  floor_geom = '<geom type="plane" pos="0 0 -0.6" size="3 3 0.1" rgba="0.5 0.5 0.5 1"/>' if floor else ""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset>{"".join(mats)}</asset>
      <worldbody>{light}{floor_geom}{"".join(geoms)}
        <camera name="cam" pos="0 -2.6 0" xyaxes="1 0 0 0 0 1" fovy="30"/>
      </worldbody>
    </mujoco>""")


def _furnace() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset>
        <material name="a" rgba="1 1 1 1" metallic="1" roughness="0.6"/>
        <material name="b" rgba="1 1 1 1" metallic="0" roughness="0.6"/>
      </asset>
      <worldbody>
        <geom type="sphere" pos="-0.3 0 0" size="0.25" material="a"/>
        <geom type="sphere" pos="0.3 0 0" size="0.25" material="b"/>
        <camera name="cam" pos="0 -2 0" xyaxes="1 0 0 0 0 1" fovy="30"/>
      </worldbody>
    </mujoco>""")


def _khronos(model: str) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(scenes.asset_dir() / "mjcf" / model / f"{model}.xml"))
  spec.visual.headlight.active = 0
  spec.worldbody.add_camera(name="cam", pos=[0, -1.7, 0.5], xyaxes=[1, 0, 0, 0, 0, 1], fovy=40)
  return spec.compile()


def _hdri(name: str) -> np.ndarray:
  return envmap_lib.load_hdr(scenes.asset_dir() / "hdri" / f"{name}_1k.hdr")


def _tubes(light: str) -> mujoco.MjModel:
  """Capsules and cylinders on a floor (no environment). The spot variant differs from Mitsuba in
  the cone falloff (ours: Filament's smooth cone, Mitsuba's: linear in angle)."""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset>
        <material name="red" rgba="0.8 0.2 0.15 1" roughness="0.4"/>
        <material name="metal" rgba="0.9 0.9 0.9 1" metallic="1" roughness="0.3"/>
        <material name="floor" rgba="0.6 0.6 0.6 1" roughness="0.8"/>
      </asset>
      <worldbody>
        {light}
        <geom type="plane" size="3 3 0.1" material="floor"/>
        <geom type="capsule" size="0.12 0.25" pos="-0.35 0 0.37" material="red"/>
        <geom type="capsule" size="0.08 0.3" pos="0.05 0.2 0.1" euler="90 0 30" material="metal"/>
        <geom type="cylinder" size="0.15 0.2" pos="0.4 0.05 0.2" material="red"/>
        <geom type="cylinder" size="0.1 0.15" pos="0.05 -0.3 0.35" euler="0 70 0" material="metal"/>
        <camera name="cam" pos="0 -2.2 1.2" xyaxes="1 0 0 0 0.5 0.866" fovy="35"/>
      </worldbody>
    </mujoco>""")


# name -> (model factory, envmap factory or None)
VALIDATION = {
  "grid_env": (lambda: _grid(""), lambda: _hdri("studio_small_09")),
  "grid_sun": (lambda: _grid('<light type="directional" dir="0.3 0.6 -0.5" intensity="3" castshadow="false"/>'), None),
  "grid_point": (lambda: _grid('<light type="point" pos="0.5 -1.5 0.8" intensity="4" range="1000" castshadow="false"/>'), None),
  "furnace": (_furnace, lambda: np.ones((64, 128, 3), np.float32)),
  "damaged_helmet": (lambda: _khronos("DamagedHelmet"), lambda: _hdri("kloofendal_48d_partly_cloudy_puresky")),
  "water_bottle": (lambda: _khronos("WaterBottle"), lambda: _hdri("venice_sunset")),
  "normal_tangent": (lambda: _khronos("NormalTangentMirrorTest"), lambda: _hdri("studio_small_09")),
  "tubes_point": (lambda: _tubes('<light type="point" pos="0.4 -0.6 1.6" intensity="3" range="1000" bulbradius="0"/>'), None),
  "tubes_spot": (
    lambda: _tubes(
      '<light type="spot" pos="0.4 -0.6 1.6" dir="-0.2 0.35 -1" intensity="6" range="1000" cutoff="35" bulbradius="0"/>'
    ),
    None,
  ),
}


def tone_map(hdr: np.ndarray, exposure: float) -> np.ndarray:
  """numpy twin of the renderer's display path: exposure, PBR Neutral, sRGB."""
  c = np.maximum(hdr * exposure, 0.0)
  x = c.min(axis=-1, keepdims=True)
  offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
  c = c - offset
  peak = c.max(axis=-1, keepdims=True)
  start, desat = 0.76, 0.15
  d = 1.0 - start
  new_peak = 1.0 - d * d / (peak + d - start)
  compressed = c * (new_peak / np.maximum(peak, 1e-9))
  g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0)
  compressed = compressed * (1 - g) + new_peak * g
  c = np.clip(np.where(peak < start, c, compressed), 0.0, 1.0)
  return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def render_ours(
  mjm, env, res: int, spp_axis: int, tier: int = mjpbr.Tier.REALTIME, path_samples: int = 256, max_depth: int = 4
):
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  kwargs = dict(envmap=env, envmap_intensity=1.0) if env is not None else {}
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

  mi.set_variant("cuda_ad_rgb")
  filament_bsdf.register(ibl.dfg_lut_array().numpy())
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
    sheet = np.concatenate(
      [a, b, np.repeat(np.asarray(flip_map)[..., None], 3, -1) if np.asarray(flip_map).ndim == 2 else flip_map], axis=1
    )
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_{args.tier}_{args.bsdf}.png")
  with (args.out / f"metrics_{args.tier}_{args.bsdf}_{args.integrator}.jsonl").open("w") as f:
    for row in rows:
      f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
  main()
