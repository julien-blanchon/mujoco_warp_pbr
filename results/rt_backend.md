# OptiX traversal backend (`traversal="optix"`): RTX PRO 6000 results

Date: 2026-10-22. Measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, RT cores)
in a Hugging Face sandbox. Driver 580.178 (OptiX 9.0 ABI), PyOptiX built against optix-dev
v9.0.0, warp-lang 1.18, Warp PTX compiled for sm_89 (the backend retries automatically). The design
is described in [`../research/web/optix_warp.md`](../research/web/optix_warp.md#6-integrated-backend-traversaloptix).

## Follow-up, 2026-10-23: primitive layout, CUDA graphs, re-framed mesh GASes

Same machine type (RTX PRO 6000 Blackwell Server Edition, driver 580.178, OptiX 9.0). The raw rows
are in [`rt_backend_v2.jsonl`](rt_backend_v2.jsonl). Frames/s; software is timed from a CUDA graph,
"OptiX before" (commit f79737f's backend) and "OptiX after" eagerly, "after, graph" from a CUDA graph
of 20 renders (`optix_graph_capture`). The ratio uses eager OptiX for batches and the graph for the
single world. All rows were measured in one sandbox session, except primitives "after" and its
software rows (second session, after the register-cap change).

| scene | worlds × res | software | OptiX before | OptiX after | after, graph | after / software |
| --- | --- | --- | --- | --- | --- | --- |
| material_grid | 4096 × 64² | 1.84M | 1.84M | 2.32M | 2.25M | 1.27× |
| material_grid | 1024 × 128² | 579k | 605k | 731k | 711k | 1.26× |
| material_grid | 1 × 256² | 36.5k | 3.2k | 3.2k | 3.7k | 0.10× |
| primitives | 4096 × 64² | 1.23M | 646k | 1.05M | 1.04M | 0.86× |
| primitives | 1024 × 128² | 341k | 189k | 306k | 302k | 0.90× |
| primitives | 1 × 256² | 28.1k | 3.3k | 2.9k | 5.3k | 0.19× |
| damaged_helmet | 4096 × 64² | 306k | 1.68M | 1.61M | 1.63M | 5.26× |
| damaged_helmet | 1024 × 128² | 103k | 475k | 467k | 456k | 4.52× |
| damaged_helmet | 1 × 256² | 5.0k | 3.6k | 3.3k | 6.7k | 1.36× |
| anymal_c | 4096 × 64² | 177k | 964k | 913k | 901k | 5.15× |
| anymal_c | 1024 × 128² | 61.8k | 310k | 308k | 303k | 4.99× |
| anymal_c | 1 × 256² | 2.4k | 3.4k | 3.2k | 4.8k | 1.99× |
| franka_showroom | 4096 × 64² | 217k | 620k | 620k | 614k | 2.86× |
| franka_showroom | 1024 × 128² | 68.2k | 221k | 250k | 246k | 3.66× |
| franka_showroom | 1 × 256² | 1.7k | 3.4k | 3.1k | 4.9k | 2.87× |
| isaac_seattle_lab_table | 4096 × 64² | 469k | 1.62M | 1.55M | 1.34M | 3.32× |
| isaac_seattle_lab_table | 1024 × 128² | 146k | 470k | 468k | 396k | 3.21× |
| isaac_seattle_lab_table | 1 × 256² | 6.2k | 3.5k | 3.3k | 6.4k | 1.02× |
| isaac_franka_cabinet | 4096 × 64² | 270k | 1.34M | 1.84M | 1.79M | 6.82× |
| isaac_franka_cabinet | 1024 × 128² | 89.9k | 380k | 535k | 517k | 5.95× |
| isaac_franka_cabinet | 1 × 256² | 1.9k | 2.8k | 3.3k | 5.1k | 2.76× |

The other score configurations (1024 × 64², 4096 × 128², helmet 2×2 spp) follow the same pattern
(rows in the jsonl). Score (geometric mean of the 13 score configurations, batched): software 423k,
OptiX before 649k, OptiX after 792k, `traversal="auto"` (software on the two primitive-only
scenes) 777k. Software numbers include commit b3a1acf (re-framed mesh BVHs: the table's software
rate went from 55k to 469k at 4096 × 64²).

**Parity** (8-bit RGB, 16 worlds, every configuration): maximum difference 0–2 levels and no
pixel above 2, except one pixel at 256² in each of material_grid, damaged_helmet and the table
(silhouette flips). `tests/test_optix.py` (16 tests, including new graph-capture, primitive
layout, re-framed GAS, reflection and `auto` tests) passes on the RTX PRO 6000 and on GH200.

### Primitives

- **Layout.** Spheres and capsules are now OptiX built-in linear curves (round linear: linear
  swept spheres, hardware-intersected on Blackwell): one world-space curve GAS, a capsule one
  segment, a sphere a zero-length segment, refitted every frame. A closest-hit program re-runs
  MuJoCo Warp's `ray_sphere` / `ray_capsule` on the hit geom, so distances and normals are the
  software path's. Boxes are instances of one unit-cube triangle GAS, scaled per box (hardware
  triangles). Only ellipsoids and cylinders keep custom intersection programs. PyOptiX 9.0 has
  no sphere build input, hence spheres as degenerate curves.
  - material_grid (49 spheres): 2.22 → 1.76 ms at 4096 × 64², now 1.27× faster than software.
  - isaac_franka_cabinet (one large box): 1.34M → 1.84M frames/s.
- **Mirror reflections were the real cost of the primitives scene.** With `has_reflection` off,
  OptiX and software take 2.37 and 2.41 ms (4096 × 64²). Its floor material has
  `reflectance=".2"`: the second closest-hit trace and the second inlined shading call site made
  the program spill at the 64-register cap (5.54 ms with the new layout). The register cap now
  defaults to 96 when the scene has reflections (64 otherwise): 3.88 ms. 80 / 112 registers:
  4.13 / 4.26 ms; 96 registers on scenes without reflections costs 3–12%.
- **Still slower than software on primitives** (0.80–0.92×). Tracing the mirror ray through the
  camera-layer loop (one trace and one shading site) matched software in the software kernel
  exactly but not in the OptiX program (bright or dark rows of floor pixels, depending on the loop
  form), so it was not kept. Instead **`traversal="auto"`** picks OptiX only when meshes are
  rendered (and OptiX is available), else software.
- The instance-per-primitive layout of the first version (4.85 vs 5.80 ms) was not measured
  again: the curve and cube layouts replace its custom intersection programs with hardware ones.

### CUDA graphs

- **Why capture fails.** Traced with Nsight Systems (GH200, OptiX 9.1), an eager `optixLaunch` is
  one `cuMemcpyDtoDAsync` (parameters) and three `cuEventRecord`s; the ray-tracing kernel is
  submitted by the driver outside the CUDA API, under the memcpy's correlation id. During capture
  the memcpy is captured but the kernel still runs at once, reading a parameter slot the captured
  copy never filled; the graph holds a single memcpy node. Capture mode (global, thread-local,
  relaxed) does not matter. Captured `optixAccelBuild`s replay correctly but break OptiX's next
  eager build ("Failed to query command list event").
- **What works: a host-function bridge** (`optix/graph.py`, opt-in with
  `rc.optix_graph_capture = True`). Inside a capture the Warp kernels of a frame are captured as
  usual; the OptiX calls are replaced by two host nodes (libc `sem_post` / `sem_wait`) and run on
  a worker thread with its own non-blocking stream. Graph replays match eager renders exactly,
  also after the bodies move (refits in the graph) and with supersampling.
- **Caveat (hence opt-in).** A synchronous copy to pageable memory (`array.numpy()`) issued while
  such a graph runs deadlocks: the copy waits for the graph while holding a driver lock that the
  worker's kernel launches need. Synchronise first (`wp.synchronize()`). Gating pre-enqueued OptiX
  work on the GPU instead (`cuStreamWaitValue32`) avoids the host on the critical path but makes
  `wp.synchronize()` wait forever for the pending work.
- **Latency.** Single world 256²: 3.2–3.6k frames/s eager → 3.7–6.7k with a graph (host cost per
  frame ~0.3 → ~0.15–0.27 ms), still far from software's 28–37k from a graph. For batches the
  graph is no faster (0–14% slower: the GPU idles during the host round trip), so bench.py keeps
  timing OptiX eagerly.

### Re-framed mesh GASes

The OptiX backend now builds triangle GASes in the authored frame where `mesh_bvh_frame="auto"`
re-frames the software BVH (instance transform `geom_xmat @ R^T`, normals unchanged). OptiX's
builder is far less sensitive to the inertia frame than Warp's: 4096 × 64², eager ms, MuJoCo frame
→ authored frame: isaac_seattle_lab_table 2.53 → 2.63, anymal_c 4.26 → 4.27, isaac_franka_cabinet
2.24 → 2.23, franka_showroom 7.20 → 6.60. It is kept (one gain, the rest within noise).

## Score preset (2026-10-22, first version)

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

- **Primitive scenes with mirror reflections** (the primitives scene) remain 0.80–0.92× of
  software; `traversal="auto"` uses software there.
- **CUDA graphs** need the opt-in host-function bridge and its synchronisation caveat (above).
- **Host cost.** The host enqueue cost is about 0.3 ms per frame, against 0.15 ms for software;
  a graph brings OptiX to ~0.15–0.27 ms per single-world frame.
- **Unsupported geometry.** Flex, height fields and SDF geoms fall back to software.
