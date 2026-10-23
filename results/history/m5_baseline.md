# M5 — real-time tier baseline (2026-10-22)

Hardware: 1× NVIDIA GH200 (Hopper, no RT cores). Warp 1.18.0, MuJoCo 3.17, mujoco_warp @ 6c33934,
Newton 1.7.0, Mitsuba 3.9.1. Raw data: `results/history/compare.jsonl`, `results/score.jsonl`,
`results/validation/`.

## Quality vs Mitsuba 3 (`benchmarks/validate.py`, 256², Mitsuba `direct` 1024 spp)

Both images go through the same exposure, PBR Neutral tone mapper and sRGB before ꟻLIP / PSNR.
"filament" renders Mitsuba with a plugin of our own BRDF (exact integral of the same material
model: the gap is the real-time approximation); "principled" uses Mitsuba's Disney BSDF (a
different material model).

| Scene | ꟻLIP (filament) | PSNR (filament) | ꟻLIP (principled) | What it tests |
| --- | --- | --- | --- | --- |
| grid_point | 0.003 | 42.7 dB | 0.076 | 5×5 metallic × roughness, point light |
| grid_sun | 0.005 | 40.3 dB | 0.096 | same, directional light |
| furnace | 0.024 | 30.1 dB | 0.061 | white metal/dielectric in a white environment |
| water_bottle | 0.032 | 35.2 dB | 0.034 | textured metal-rough asset, HDRI |
| normal_tangent | 0.088 | 21.9 dB | 0.094 | normal maps incl. mirrored UVs, HDRI |
| damaged_helmet | 0.096 | 23.8 dB | 0.093 | full texture set + emission, HDRI |
| grid_env | 0.116 | 23.4 dB | 0.130 | 5×5 grid under a studio HDRI only |

- Direct lighting is exact up to anti-aliasing and Monte-Carlo noise (ꟻLIP ≤ 0.005).
- Under image-based lighting each grid sphere's mean radiance is within ~3% of the exact
  integral (rough metal +5%); the remaining ꟻLIP comes from what the real-time tier does not
  model: environment visibility (spheres and helmet parts occluding the sky) and split-sum
  approximations. Texture AO (ours) vs true occlusion (Mitsuba) adds to the helmet gap. The
  path-traced tier (M6) removes these.
- Mitsuba's principled BSDF loses energy at high roughness and uses Burley diffuse; Filament's
  model (ours) is energy-compensated, hence the larger gaps in the "principled" column.

## Speed (`benchmarks/sweep.py --preset compare`, since replaced by `benchmarks/suite.py`; CUDA graph, median of 5 × 20 frames)

Milliseconds per batch (one 64² or 128² camera per world). Ours: PBR with HDRI, ray-traced
shadows, textures, specular AA. mujoco_warp: its renderer with shadows on (Lambert + Phong).
Newton: `SensorCamera` with its benchmark settings (Lambert + ambient, shadows, textures).

| Scene | nworld | res | ours | mujoco_warp | Newton |
| --- | --- | --- | --- | --- | --- |
| material_grid | 1024 | 64 | 1.45 | 0.69 | 1.49 |
| material_grid | 4096 | 64 | 5.82 | 2.75 | 6.20 |
| material_grid | 1024 | 128 | 5.85 | 2.46 | 4.04 |
| primitives | 1024 | 64 | 5.97 | 1.57 | 1.68 |
| primitives | 4096 | 64 | 24.04 | 6.32 | 6.99 |
| primitives | 1024 | 128 | 21.88 | 5.56 | 4.87 |
| damaged_helmet | 1024 | 64 | **7.94** | 9.49 | 7.87 |
| damaged_helmet | 4096 | 64 | **31.40** | 37.74 | 31.06 |
| damaged_helmet | 1024 | 128 | 28.69 | 35.44 | 21.65 |
| flight_helmet | 1024 | 64 | **10.18** | 11.81 | 11.01 |
| flight_helmet | 4096 | 64 | **40.27** | 46.85 | 42.93 |
| flight_helmet | 1024 | 128 | 34.39 | 46.28 | 28.76 |

MuJoCo's OpenGL renderer (EGL, one image at a time, incl. readback): 350–590 frames/s at 64².

Observations:
- Mesh scenes: PBR at mujoco_warp's speed or better, because the megakernel compiles with
  `max_unroll=6` (mujoco_warp's mesh query wraps `wp.mesh_query_ray` in an 8-iteration loop that
  Warp's default unrolling inlines 16 times; rolling it is 25% faster). mujoco_warp would gain the
  same from this change.
- primitives: the grid floor has MuJoCo `reflectance=0.2`, so every floor pixel traces and shades a
  mirror ray (+2 ms); the other renderers ignore reflectance. Texture UV derivatives by finite
  differences cost a further ~0.8 ms.
- Newton is fastest at 128²: its 8×8 tiled thread mapping keeps rays coherent — the first
  optimisation candidate.
- Throughput scales linearly in nworld (1024 → 4096) for all renderers.

## Optimisation score (`benchmarks/sweep.py --preset score`)

Geometric mean camera-frames/s over 3 scenes × nworld {1024, 4096} × res {64, 128} at 1 spp plus a
2×2-spp point, all with an HDRI: **113.9k frames/s** (`m5-final`; 98.6k before the unroll fix).
This is the starting point for M7.
