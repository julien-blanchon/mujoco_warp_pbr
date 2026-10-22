# mujoco_warp_pbr

A batched, physically based renderer for [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
written in [NVIDIA Warp](https://github.com/NVIDIA/warp). It renders thousands of MuJoCo Warp worlds
per launch with a single specialised megakernel, using MuJoCo's own material, texture and light
fields.

- **Real-time tier** (default): primary ray, ray-traced (soft) shadows, Filament's BRDF with
  multiscatter energy compensation, split-sum image-based lighting from HDR environment maps,
  normal / ORM / emissive / opacity maps with ray-cone mip selection, specular anti-aliasing,
  in-kernel supersampling, MuJoCo reflectance (mirror ray) and alpha blending. Deterministic and
  noise-free.
- **Path-traced tier**: unidirectional path tracing with next-event estimation and multiple
  importance sampling (lights, environment, BSDF). Same BRDF, converges to Mitsuba 3. Optional
  edge-avoiding à-trous denoiser (`denoise=True`) guided by first-hit albedo, normal and depth.
- Every material, light and environment parameter can vary per world (domain randomisation)
  without recompiling. Depth and segmentation match MuJoCo Warp's renderer.

## Usage

```python
import mujoco
import mujoco_warp as mjw
import warp as wp

import mujoco_warp_pbr as mjpbr

mjm = mujoco.MjModel.from_xml_path("scene.xml")
mjd = mujoco.MjData(mjm)
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=1024)

rc = mjpbr.create_render_context(
  mjm,
  nworld=1024,
  cam_res=(128, 128),
  render_rgb=True,
  render_depth=True,
  envmap="studio.hdr",  # optional equirectangular HDR environment
)

mjw.step(m, d)
mjw.refit_bvh(m, d, rc.base)  # after the geometry moved
mjpbr.render(m, d, rc)

rgb = wp.zeros((1024, 128, 128), dtype=wp.vec3)
mjpbr.get_rgb(rc, 0, rgb)  # camera 0, display RGB in [0, 1]
```

Per-world variation: replace any batched array of the context or model with one whose leading
dimension is `nworld`, e.g. `rc.geom_roughness = wp.array(r, dtype=float)` with `r` of shape
`(nworld, ngeom)`, `rc.env_id` to select one of several environment maps per world, or
`m.geom_rgba` (MuJoCo Warp's convention: arrays are indexed with `worldid % shape[0]`).

## Materials and lights

Only native MuJoCo fields are used, so existing MJCF works unchanged:

| MuJoCo | Renderer |
| --- | --- |
| `rgba`, texture layer `rgb` / `rgba` | base colour (linear) and alpha |
| `metallic`, `roughness` (-1 = unset) | metallic, perceptual roughness; unset: metallic 0, roughness 1 − `shininess` (or `unset_material=UnsetMaterial.HEURISTIC`, or `default_material=`) |
| `specular` | dielectric reflectance, F0 = 0.16 · specular² (0.5 → 0.04) |
| layers `orm`, `occlusion`, `roughness`, `metallic` | ambient occlusion, roughness and metallic maps (factor × texel) |
| layer `normal` | tangent-space normal map (glTF convention) |
| `emission`, layer `emissive` | emitted radiance (display-relative: 1 = full brightness) |
| layer `opacity` | alpha |
| `reflectance` | mirror reflection blended by reflectance |
| light `intensity` (lux / candela), `range`, `bulbradius`, `diffuse` | physical lights (`light_model=LightModel.PHYSICAL`, default); intensity 0 uses a calibrated fallback |
| light `attenuation`, `cutoff`, `exponent`, `ambient`, `specular` | MuJoCo's GL semantics (`light_model=LightModel.MUJOCO`) |
| skybox texture | environment lighting and background when no `envmap` is given |

Two converters produce MJCF with these fields:
- `tools/gltf_to_mjcf.py` converts glTF 2.0 assets (metallic-roughness core);
- `tools/usd_to_mjcf.py` converts USD stages: UsdPreviewSurface and OmniPBR materials with their
  textures, and UsdLux lights mapped to MuJoCo lights. It is used for the Isaac Sim / IsaacLab
  demo scenes in `benchmarks/scenes.py`.

Supported geometry: planes, spheres, capsules, ellipsoids, cylinders, boxes, meshes, height fields
and SDFs with full PBR; flex is shaded without normal maps and Gaussian splats keep MuJoCo Warp's
unlit compositing (a warning is printed for both).

## Newton

[`newton-pbr/`](newton-pbr/README.md) is a small package depending on this one and on
[Newton](https://github.com/newton-physics/newton). Its `newton_pbr.SensorPbrCamera` mirrors
`newton.sensors.SensorCamera.update`, so any Newton scene renders with this renderer, batched over
all worlds:

```python
camera = newton_pbr.SensorPbrCamera(model, width=128, height=128, fovy=60.0, envmap="studio.hdr")
camera.update(state, camera_transforms, color_image=color, depth_image=depth)
```

## Performance

On one GH200 (Hopper, no RT cores), with an HDR environment and shadows, the real-time tier
renders the DamagedHelmet scene at ~174k frames/s (4096 worlds, 64²), 1.6× MuJoCo Warp's
Lambert/Phong renderer. On primitive-only scenes it runs at 0.5–0.7× MuJoCo Warp's speed, where
PBR shading outweighs traversal. Throughput saturates from about 1–4 M pixels per launch (e.g.
256 worlds at 64²). See `results/scaling.md`, `results/m5_baseline.md` and
`results/m7_autoresearch.md` for tables, and `results/m6_pathtrace.md` for the path-traced tier.

Tuning knobs: `max_registers` (96 by default, tuned on Hopper; 0 leaves it to the compiler),
`render_order` (`RenderOrder.TILED` default; `RenderOrder.WORLD` is up to 2.7×
faster when all worlds render nearly the same image, e.g. fixed cameras), `block_dim`,
`samples_per_pixel`, `shadow_samples`, and disabling unused features (`use_shadows`,
`use_textures`, `specular_aa`, `enable_reflections`).

## Benchmarks and validation

```bash
python tools/fetch_assets.py            # Khronos glTF, Menagerie robots, Poly Haven HDRIs
python benchmarks/bench.py --scene damaged_helmet --nworld 4096 --res 64
python benchmarks/sweep.py --preset score --out results/score.jsonl   # also: scaling, tune, renderers
python benchmarks/validate.py --res 256 --spp 1024   # FLIP / PSNR against Mitsuba 3
python benchmarks/gallery.py --scene franka_showroom --frames 48      # side-by-side orbit video
```

Assets are cached in `$MJPBR_ASSET_DIR` (default `~/.cache/mujoco_warp_pbr`). The reference
renderers (Mitsuba 3, Newton) and image metrics install with the `bench` extra.
`tools/fetch_assets.py --only isaac` fetches the Isaac demo scenes; their licence allows local use
only.

`benchmarks/suite.py` compares every renderer:
- this renderer: real-time tier, path-traced tier, and through `newton_pbr`;
- MuJoCo Warp, MuJoCo's OpenGL renderer and Mitsuba 3;
- Newton's `SensorCamera` and `ViewerRTX` (NVIDIA Omniverse RTX, on RTX GPUs).

It runs them on showcase, robotics and light-rig scenes, for a single scene and for batches, and
writes a markdown report with orbit videos. `tools/hf/run_suite.py` runs it on several
[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) GPUs, with a private
bucket carrying code, assets and results:

```bash
uv run tools/hf/run_suite.py --bucket USER/BUCKET --flavors rtx-pro-6000 h200 a100-large --sync-assets --wait
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Design notes and literature: [`research/`](research/README.md).
