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

## Install

```bash
pip install mujoco-warp-pbr          # Python >= 3.12; pulls mujoco, mujoco-warp and warp-lang
pip install newton-pbr               # optional: Newton camera sensor and viewer
pip install "mujoco-warp-pbr[typing]"  # optional: shape-aware type stubs for Pyrefly
```

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

Camera rays: by default each camera is a pinhole (MuJoCo's `fovy` / intrinsics). Pass per-pixel
camera-frame rays instead (distorted or fisheye lenses, any projection) with
`create_render_context(..., camera_rays=rays)` or `mjpbr.set_camera_rays(rc, rays)`: one
`(height, width, 2)` vec3 array per rendering camera, origins in `[..., 0]` and directions in
`[..., 1]` (-z forward, +y up, the layout of Newton's `camera_rays`). Both tiers and the OptiX
traversal use them; supersampling and the path tracer's sub-pixel jitter interpolate between
neighbouring rays (exactly for pinhole bundles) and texture filtering estimates the pixel
footprint from them. With rays, depth is the distance along each ray. The mode is compiled in:
the kernels of pinhole cameras do not contain it and render bit-identically. `render_guides=True`
also writes each pixel's first-hit base colour and world-space normal (`get_albedo` /
`get_normal`).

On GPUs with RT cores, `traversal="optix"` traces the real-time tier's rays with OptiX instead of
Warp's software BVHs: 3–7× faster on mesh scenes, at most 1.3× on primitive-only scenes (and
slower with a mirror floor); `traversal="auto"` picks OptiX only when meshes are rendered. It needs
PyOptiX
([otk-pyoptix](https://github.com/NVIDIA/otk-pyoptix), built against the OptiX version of the
driver) and the OptiX SDK headers (`OPTIX_INCLUDE_DIR`), and falls back to the software path with
a warning otherwise; see `results/rt_backend.md` for what it supports.

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

`tools/gltf_to_mjcf.py` converts glTF 2.0 assets (metallic-roughness core) to MJCF with these
fields.

Supported geometry: planes, spheres, capsules, ellipsoids, cylinders, boxes, meshes, height fields
and SDFs with full PBR; flex is shaded without normal maps and Gaussian splats keep MuJoCo Warp's
unlit compositing (a warning is printed for both).

## Newton

[`newton-pbr/`](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/newton-pbr/README.md) is a small package depending on this one and on
[Newton](https://github.com/newton-physics/newton). It provides drop-in replacements for Newton's
camera sensor and GL viewer: switch with one import or one constructor.

```python
from newton_pbr import SensorCamera  # instead of: from newton.sensors import SensorCamera

camera = SensorCamera(model, envmap="studio.hdr")  # Newton's constructor, plus PBR options
rays = SensorCamera.compute_camera_rays_pinhole(640, 480, camera_fov=1.0)  # or OpenCV / fisheye
color = camera.allocate_image_color(view_count, 640, 480)
camera.update(state, camera_transforms, rays, color_image=color)  # Newton's update()

viewer = newton_pbr.ViewerPbr()  # instead of: newton.viewer.ViewerGL()
```

`SensorCamera` keeps Newton's outputs (colour, ray / forward depth, shape index, normal, albedo,
HDR), view-to-world mapping, clear values and render settings; `ViewerPbr` keeps ViewerGL's window,
UI and camera controls and renders the main view (or, headless, `get_frame()`) with this
renderer. See the package README for what differs from Newton (deformables, particles and
Gaussians are not rendered).

## Performance

Camera-frames per second, 4096 worlds at 64², one camera per world, HDR environment and shadows
for this renderer (RTX PRO 6000 Blackwell; from `benchmarks/suite.py`, full tables for three GPUs
and all renderers in [`results/suite/suite-v2`](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/results/suite/suite-v2/README.md)):

| scene | real-time | real-time (OptiX) | Newton `SensorCamera` | Newton RTX (tiled) | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: |
| DamagedHelmet | 312k | **1.61M** | 168k | 71k | 112k |
| ABeautifulGame (1.5M triangles) | 246k | **1.33M** | 134k | 65k | 147k |
| ANYmal C | 180k | **926k** | 176k | 73k | 82k |
| Franka showroom (authored lights) | 226k | **795k** | 285k | 72k | 78k |
| material grid (spheres) | 1.87M | 2.39M | 1.95M | 73k | **2.54M** |
| primitives | 1.22M | 1.05M | 1.88M | 73k | **2.08M** |

- Newton's `SensorCamera` and MuJoCo Warp shade with albedo and simple lights (Newton's MJCF
  import also drops the scenes' lights and textures); this renderer runs full PBR with
  image-based lighting.
- On primitive-only scenes it runs at 0.6–0.7× MuJoCo Warp's speed, where PBR shading outweighs
  traversal.
- Across GPUs (geometric mean over the suite's scenes, 4096 × 64², real-time tier without
  OptiX): RTX PRO 6000 418k, H200 167k, A100 115k frames/s.
- Single-camera renderers for reference (frames/s, 256², DamagedHelmet): MuJoCo's OpenGL
  renderer 4.4k, Newton's `ViewerRTX` 91, Mitsuba 3 at 16 spp 218, this renderer's path-traced
  tier at 16 spp 82 (5.0k real-time; measured before the path tier applied `max_registers`, which
  made it about 2× faster). Batched through one tiled render product (the "Newton RTX
  (tiled)" column, static scenes only, `research/web/ovrtx_batching.md`), Omniverse RTX runs at
  about 70k frames/s whatever the scene.
- Throughput saturates from about 1–4 M pixels per launch (e.g. 256 worlds at 64²).

![DamagedHelmet in every renderer](https://raw.githubusercontent.com/julien-blanchon/mujoco_warp_pbr/main/results/suite/suite-v2/rtx-pro-6000/gallery/damaged_helmet.png)

Left to right: MuJoCo OpenGL, MuJoCo Warp, Newton `SensorCamera`, Newton `ViewerRTX`, this
renderer's real-time tier, its path-traced tier (256 spp) and Mitsuba 3 (256 spp).

See `results/scaling.md` and the milestone logs in [`results/history/`](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/results/history/README.md)
(`m5_baseline.md`, `m7_autoresearch.md`) for more tables, `results/history/m6_pathtrace.md` for the
path-traced tier and `results/rt_backend.md` for the OptiX backend.

Tuning knobs: `max_registers` (96 by default, tuned on Hopper; 0 leaves it to the compiler; both
tiers apply it), `render_order` (`RenderOrder.TILED` default; `RenderOrder.WORLD` is up to 2.7×
faster when all worlds render nearly the same image, e.g. fixed cameras), `block_dim`,
`samples_per_pixel`, `shadow_samples`, and disabling unused features (`use_shadows`,
`use_textures`, `specular_aa`, `enable_reflections`). The host side of a `render()` call takes
about 61 µs (was 136 µs).

Settings are compiled into the kernel only where reading them at runtime measured slower
(DamagedHelmet and the material grid, 4096 worlds at 64², RTX PRO 6000):

| compiled-in setting | why | measured runtime cost |
| --- | --- | ---: |
| `samples_per_pixel` | sample loop | 4–20% |
| translucency (`max_layers`) | layer loop | 2–10% |
| reflections, textures, environment map, light model, denoise | code paths | 2–6% |
| light count, `render_order` | unrolled loop / pixel mapping | 1–4% |
| tier, geom types, traversal, fast math, `max_registers` | structural | – |

Changing one of these recompiles the kernel at the next render (about 2–7 s on GPU, 2 s on CPU);
every other setting (tone mapper, shadows and shadow samples, headlight, clip planes, HDR output,
environment background, specular AA, vertex normals, spot lights, …) is read at runtime.

## Running on CPU

No GPU is needed: the renderer runs unchanged on Warp's CPU device, with images within 1/255 of
the GPU's for the real-time tier.

```python
with wp.ScopedDevice("cpu"):
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=16)
  rc = mjpbr.create_render_context(mjm, nworld=16, cam_res=(64, 64), render_rgb=True)
  mjpbr.render(m, d, rc)
```

Warp runs a CPU launch on a single core, so throughput does not grow with `nworld`. To use every
core, give each core its own worker, with its own `Data` and render context: Python threads
(Warp's launch releases the GIL) or pinned processes. `benchmarks/cpu_bench.py` measures this,
plus MuJoCo's OpenGL renderer on a software rasteriser and Mitsuba 3's LLVM backend:

```bash
python benchmarks/cpu_bench.py --scene damaged_helmet --res 64 --nworld 32 --workers 1 8 32 [--mode process]
python benchmarks/cpu_bench.py --parity --scene damaged_helmet --res 64   # CPU vs GPU images
```

DamagedHelmet at 64² renders at about 125–150 frames/s per core, scaling to 2.5k frames/s on a
16-core / 32-thread EPYC and 34k on a GH200 node (4× 72-core Grace). One GH200 GPU is 6–47×
faster than one 72-core Grace. A render context with an HDR environment takes about 2 s to
create on the CPU device (the map's prefilter runs in row bands on up to 16 threads; it took
about 32–35 s on one core). Full tables, the scaling curve and the baselines
are in [`results/cpu.md`](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/results/cpu.md).

## Benchmarks and validation

```bash
python tools/fetch_assets.py            # Khronos glTF, Menagerie robots, Poly Haven HDRIs
python benchmarks/bench.py --scene damaged_helmet --nworld 4096 --res 64
python benchmarks/bench.py --scene damaged_helmet --nworld 4096 --res 64 --gpus 0 1 2 3   # multi-GPU scaling
python benchmarks/sweep.py --preset score --out results/score.jsonl   # also: scaling, tune
python benchmarks/validate.py --res 256 --spp 1024   # FLIP / PSNR against Mitsuba 3
python benchmarks/denoise_eval.py --scene grid_env --samples 4 16   # FLIP of the denoiser
python benchmarks/gallery.py --scene franka_showroom --frames 48      # side-by-side orbit video
```

Assets are cached in `$MJPBR_ASSET_DIR` (default `~/.cache/mujoco_warp_pbr`); `--set
envmap=hdri/<file>.hdr` and the other scripts resolve relative environment maps against it. The
reference renderers (Mitsuba 3, Newton) and image metrics install with the `bench` extra.
`benchmarks/cpu_bench.py` is the CPU counterpart of `bench.py` (see Running on CPU).

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
uv run pytest -m "not slow"   # skips the high-sample path-traced references
uv run ruff check . && uv run ruff format --check .   # or: pre-commit install
uv run pyrefly check
```

**Type checking.** [Pyrefly](https://pyrefly.org/en/docs/tensor-shapes/) is the type checker. The `dev` extra
pulls in Pyrefly and the `typing` extra (`pip install mujoco-warp-pbr[typing]`): shape-aware stubs for Warp
(`pyrefly-warp-stubs`) and NumPy (`pyrefly-numpy-stubs`), and `mujoco-stubs`. Host code annotates array shapes
with named dimensions, e.g. `wp.array[wp.vec3, [NWorld, H, W]]` or `np.ndarray[[H, W, 3], np.dtype[np.float32]]`,
so image sizes, batch and channel counts are checked across calls; kernel signatures stay plain Warp
(`wp.array2d[float]`). The annotations are type-checking-only (`from __future__ import annotations`), so a plain
install needs neither Pyrefly nor the stubs. `uv sync --extra dev` is enough for `pyrefly check`: the
benchmark-only packages (`bench` extra, OptiX, ovrtx, huggingface_hub) are listed in `ignore-missing-imports`, so
their imports are untyped when absent and checked when installed.

Design notes and literature: [`research/`](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/research/README.md).
