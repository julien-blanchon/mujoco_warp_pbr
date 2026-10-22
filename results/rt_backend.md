# OptiX traversal backend (`traversal="optix"`): RTX PRO 6000 results

Date: 2026-10-22. Measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, RT cores)
in a Hugging Face sandbox. Driver 580.178 (OptiX 9.0 ABI), PyOptiX built against optix-dev
v9.0.0, warp-lang 1.18, Warp PTX compiled for sm_89 (the backend retries automatically). The design
is described in [`../research/web/optix_warp.md`](../research/web/optix_warp.md#6-integrated-backend-traversaloptix).

## Score preset

This is `python benchmarks/sweep.py --preset score` with and without
`--extra --set 'traversal="optix"'`. The raw rows are in [`rt_backend.jsonl`](rt_backend.jsonl).

Software rows are timed from a CUDA graph. OptiX rows are timed eagerly: stream capture does not
record OptiX launches, so bench.py's `time_eager` enqueues the iterations back to back and then
synchronises. For the software path, eager and graph timings agree within 2% (`time_eager` vs
`time_graph`, measured on all nine 1-spp configurations).

| scene | worlds × res | spp | software ms | OptiX ms | speed-up |
| --- | --- | --- | --- | --- | --- |
| material_grid | 1024 × 64² | 1 | 0.56 | 0.67 | 0.84× |
| material_grid | 1024 × 128² | 1 | 1.75 | 1.62 | 1.08× |
| material_grid | 4096 × 64² | 1 | 2.20 | 2.17 | 1.02× |
| material_grid | 4096 × 128² | 1 | 7.14 | 5.91 | 1.21× |
| primitives | 1024 × 64² | 1 | 0.85 | 1.70 | 0.50× |
| primitives | 1024 × 128² | 1 | 2.97 | 5.40 | 0.55× |
| primitives | 4096 × 64² | 1 | 3.32 | 6.10 | 0.54× |
| primitives | 4096 × 128² | 1 | 11.87 | 20.42 | 0.58× |
| damaged_helmet | 1024 × 64² | 1 | 3.43 | 0.71 | 4.86× |
| damaged_helmet | 1024 × 128² | 1 | 9.94 | 2.15 | 4.62× |
| damaged_helmet | 4096 × 64² | 1 | 12.88 | 2.47 | 5.22× |
| damaged_helmet | 4096 × 128² | 1 | 39.00 | 8.19 | 4.76× |
| damaged_helmet | 1024 × 64² | 2×2 | 13.71 | 2.43 | 5.64× |
| **score** (geometric mean of frames/s) | | | **426k** | **662k** | **1.55×** |

## Other scenes (eager timing, 64², 1 spp)

| scene | geoms | worlds | software ms | OptiX ms | speed-up |
| --- | --- | --- | --- | --- | --- |
| anymal_c | 50 meshes, 44 primitives, plane | 1024 | 6.14 | 1.16 | 5.3× |
| franka_showroom | 71 meshes, 10 boxes, plane | 1024 | 5.82 | 1.87 | 3.1× |
| flight_helmet | 6 meshes (95k triangles), plane | 1024 | 3.49 | 0.75 | 4.6× |

## Parity with the software traversal

- **RGB.** Measured on every score configuration, comparing the 8-bit outputs of both traversals.
  The maximum difference is 1–2 levels, with 0 pixels differing by more than 2. The mean absolute
  difference is ≤ 0.009 levels.
- **Depth and segmentation.** The test scene has primitives of every type, a rotated mesh, a plane
  and three worlds:
  - segmentation is identical on every pixel and depth differs by at most 1.1e-5;
  - `tests/test_optix.py` also moves free bodies per world, and requires ≥ 99.8% identical
    segmentation and depth within rtol 2e-4.
- **Test suite.** `tests/test_optix.py` passes on the RTX PRO 6000 and on GH200, where OptiX
  traverses in software.

## Settings tried

- **Register cap (`optix_max_registers`), 4096 × 64².** Default 64.
  - OptiX's default of 0 gives 2.68 ms on material_grid and 3.09 ms on damaged_helmet.
  - A cap of 48–64 gives 2.12–2.17 ms and 2.51 ms.
  - 128 and 168 are slower, up to 3.95 ms on damaged_helmet.
- **Primitive layout**, primitives / material_grid at 4096 × 64²:
  - one IAS instance per primitive geom over a shared, scaled unit-cube GAS: 4.85 / 2.68 ms at
    OptiX's default register count, and material_grid 2.16 ms at the 64-register cap;
  - one world-space custom-primitive GAS over all worlds (kept, since it was measured on the
    mixed scenes): 5.80 ms on primitives at the default register count, and 6.19 / 2.21 ms at
    the cap. The instance layout is therefore faster on capsules and is the first follow-up;
  - hybrid, with primitives in MuJoCo Warp's BVH inside the OptiX program: 8.71 / 2.71 ms. It is
    also 2.6× slower on damaged_helmet, so it was dropped;
  - tight world-space bounds for rotated capsules and cylinders: no change (6.29 ms).
- **Supersampling.** With trace calls inside the sub-pixel sample loop, damaged_helmet at 2×2 spp
  took 42.7 ms. The OptiX compiler reported 2160 B of direct spills and 4651 B of continuation
  spills, against 588 B at 1 spp. Rendering the n×n samples as an n×-resolution image (one launch
  index per sample) and then downsampling brings it to 2.43 ms. Unrolling the loop was worse
  (70 ms on GH200).

## Open issues

- **Primitive-heavy scenes.** The primitives scene (25 capsules and 25 spheres) is about 1.8×
  slower than software. Its intersection programs run on the SMs, and OptiX's per-call overhead
  outweighs the RT-core traversal. Spheres alone (material_grid) break even.
- **No CUDA graph capture.** `render` raises inside a stream capture.
- **Host cost.** The host enqueue cost is about 0.3 ms per frame, against 0.15 ms for software:
  five extra launches plus two acceleration-structure builds.
- **Unsupported geometry.** Flex, height fields and SDF geoms fall back to software.
