# Newton Warp ray tracer & IsaacLab renderers

## Newton (newton-physics/newton)
- `newton/_src/sensors/sensor_camera_render/` (active, `SensorCamera`) and `warp_raytrace/`
  (frozen, deprecated `SensorTiledCamera`). Sibling lineage of mjwarp's renderer (same
  megakernel naming, `wp.static` specialisation, grouped `wp.Bvh`, packed RGBA, identical
  shadow-ray comments/constants).
- Megakernel per config (`create_kernel()`, `module="unique"`); depth/shape-only fast variant.
- **Thread mapping** (`tiling.py`, `RenderOrder`): `PIXEL_PRIORITY` (view index fastest ⇒
  adjacent threads = same pixel, different worlds), `VIEW_PRIORITY`, `TILED` (default 16×8,
  benchmarks use 8×8). `block_dim` default 64 (launch-only, excluded from cache key).
- AA: `SSAA` and **`MSAA`** (trace all sub-rays, shade each distinct surface once, 4 slots/px).
- BVH: one `wp.Bvh` (lbvh/sah/cubql) over all worlds with groups; second query for global
  shapes; nested `bvh_query_ray` aliasing worked around for Gaussians (`raytrace.py:170`).
- Shading: albedo × hemispheric ambient + Σ N·L·attenuation (`1/(1+0.02d²)`), fixed spot
  smoothstep, hard shadows keep 30%. No PBR/specular/IBL/tone mapping. sRGB-aware.
- Outputs: color, hdr_color, depth, forward_depth, normal, albedo, shape_index; fisheye models.
- Bench: `asv/benchmarks/simulation/bench_sensor_camera.py` (Franka-cabinet, ANYmal, 256 shapes).

## IsaacLab
- `scripts/benchmarks/benchmark_renderer.py`: `OVRTX_RENDERER="ovrtx_renderer"`,
  `NEWTON_RENDERER="newton_renderer"`; Franka-cabinet render benchmark, reports ms + Gpx/s.
- **OVRTX** (`source/isaaclab_ov/.../renderers/`): Omniverse RTX without Kit (`ovrtx==0.5.1`),
  one tiled RenderProduct atlas for all envs, DLPack, default `RealTimePathTracing`, or
  `Minimal` modes (constant diffuse / diffuse MDL / full MDL). Needs RT-core RTX GPUs.
  Known issue: >4 GiB atlas ⇒ silent black images.
- **Newton renderer** (`isaaclab_newton/renderers/newton_warp_renderer.py`) wraps the deprecated
  `SensorTiledCamera`, TILED 8×8, block 64, CUDA graph.
- Reported (`docs/source/_static/benchmarks/environment-performance-release.csv`, RTX PRO 6000
  Blackwell, 8192 envs, 64×64, training FPS): Cartpole-Camera Newton 140.6k vs OVRTX 91.8k vs
  Isaac RTX 67.2k; Shadow-Hand 118.0k / 72.7k / 34.2k; KukaAllegro Lift 75.9k / 57.1k / 12.7k.
- Platform: IsaacLab supports Linux aarch64 (DGX Spark); RTX renderers need RT cores.

## Takeaways
- Newton `SensorCamera` = extra pure-Warp speed baseline runnable on GH200.
- Ideas: `RenderOrder` mapping sweep (incl. world-fastest), MSAA shade-once, CUDA graph capture.
