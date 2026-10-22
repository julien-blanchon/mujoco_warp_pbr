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

`tools/gltf_to_mjcf.py` converts glTF 2.0 assets (metallic-roughness core) into MJCF with these
fields.

Supported geometry: planes, spheres, capsules, ellipsoids, cylinders, boxes, meshes, height fields
and SDFs with full PBR; flex is shaded without normal maps and Gaussian splats keep MuJoCo Warp's
unlit compositing (a warning is printed for both).

## Performance

On one GH200 (Hopper, no RT cores), 4096 worlds at 64² with an HDR environment, the real-time
tier renders mesh scenes at or above the speed of MuJoCo Warp's Lambert/Phong renderer and of
Newton's camera sensor, e.g. ~130k frames/s on the DamagedHelmet scene. See
`results/m5_baseline.md` and later reports for tables, and `results/m6_pathtrace.md` for the
path-traced tier.

Tuning knobs: `max_registers` (96 by default, tuned on Hopper; 0 leaves it to the compiler),
`render_order` (`RenderOrder.TILED` default; `RenderOrder.WORLD` is up to 2.7×
faster when all worlds render nearly the same image, e.g. fixed cameras), `block_dim`,
`samples_per_pixel`, `shadow_samples`, and disabling unused features (`use_shadows`,
`use_textures`, `specular_aa`, `enable_reflections`).

## Benchmarks and validation

```bash
python tools/fetch_assets.py            # Khronos glTF samples (converted) + Poly Haven HDRIs
python benchmarks/bench.py --scene damaged_helmet --nworld 4096 --res 64
python benchmarks/sweep.py --preset compare --out results/compare.jsonl
python benchmarks/validate.py --res 256 --spp 1024   # FLIP / PSNR against Mitsuba 3
```

Assets are cached in `$MJPBR_ASSET_DIR` (default `~/.cache/mujoco_warp_pbr`). The reference
renderers (Mitsuba 3, Newton) and image metrics install with the `bench` extra.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Design notes and literature: [`research/`](research/README.md).
