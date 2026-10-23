# OptiX from Warp: feasibility for an RT-core backend

Date: 2026-10-22.

**Setup.** GH200 (sm_90, no RT cores; OptiX runs its traversal on the SMs), driver 610.57,
OptiX 9.1 headers (github.com/NVIDIA/optix-dev), PyOptiX 9.1.0 built from
github.com/NVIDIA/otk-pyoptix, warp-lang 1.18.0. Proof-of-concept code is in
[`../optix_poc/`](../optix_poc/README.md).

## 1. Warp kernels as OptiX programs: works

Warp 1.18 ships an experimental external-compilation API designed with OptiX as its first
consumer (design doc: `warp/design/external-compilation-extensions.md`):
- `wp.build_experimental.add_builtin` / `add_native_type`, and `wp.ModuleBuildOptions` for
  include directories and a CUDA preamble;
- `@wp.kernel(entry_point_abi="external_constant_params", name="__raygen__x")`: an
  `extern "C"` entry point reading one struct from the `__constant__` symbol `params`, which is
  OptiX's launch-parameter convention;
- `wp.compile_aot_module(use_ptx=True, strip_hash=True)`, which returns the PTX paths.

**What was verified:**
- Raygen, miss, closest-hit, any-hit and intersection programs can all be written as Warp kernels.
- A 256² depth image over an IAS of the DamagedHelmet triangle GAS plus a custom-sphere GAS:
  - hit codes agree 100% with a Warp reference (`mesh_query_ray` plus analytic spheres);
  - max |Δt| is 5e-4;
  - the OptiX validation layer reports no errors.
- An any-hit cutout program works.
- `wp.texture_sample` and `wp.mesh_query_ray` work inside a raygen, so hybrid software/RT
  traversal is possible.

**Pattern:**
- **Device glue.** `optixTrace` is variadic and takes payloads by reference, so it can't be
  registered as a builtin directly. A small header wraps it with fixed payload counts that are
  returned packed (e.g. `vec2(t, prim)`), plus one-liners for
  `optixGetLaunchIndex`, `optixSetPayload_N`, `optixGetRayTmax`, primitive index, barycentrics,
  object-space ray and report/ignore intersection. These are registered with `add_builtin`, and
  the header is included through `extra_cuda_preamble`.
- **Programs.** All programs in a module take the same single `@wp.struct` (OptiX `params`), which
  holds the traversable handle, the Warp output arrays and the scene data.
- **Indexing.** `wp.tid()` is replaced by the launch-index builtin.
- **Host side.** PyOptiX creates the context on Warp's CUDA context and builds the pipeline (launch
  params variable `params`, SBT). The params buffer holds the struct's raw bytes, and launches
  run on Warp's stream.
- **Instances.** `OptixInstance` (80 B) can be a `@wp.struct`, so a Warp kernel writes the instance
  transforms each step.

**Gotchas:**
- Include only `<optix_device.h>`, with `#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 0`, because
  `<optix.h>` needs `cuda_fp16.h` / `cuda.h`, which Warp's NVRTC setup lacks.
- With `strip_hash=True`, AOT compilation silently reuses an existing PTX file of the same name in
  `module_dir`. Compile into a fresh directory.
- Kernel-level `module_options=` requires `module="unique"`. Use `wp.set_module_options` instead.
- The entry-point ABI forbids `wp.tid()`, shared-memory tiles, backward passes and multiple
  arguments.
- OptiX build-input arrays must stay alive until the build completes.

## 2. Batching thousands of worlds: one IAS over all worlds

Measured with 4096 worlds and a shared DamagedHelmet GAS (15k triangles). Median of 7 runs on an
idle GPU.

| layout | 2 instances/world (8k) | 10/world (41k) | 50/world (205k) |
| --- | --- | --- | --- |
| one IAS over all worlds, refit | 0.04–0.05 ms | 0.065 ms | 0.17 ms |
| same, full rebuild (fast-build / fast-trace) | 0.20 / 0.41 ms | 0.26 / 0.59 ms | 0.49 / 0.91 ms |
| one IAS per world, refit (4096 builds) | 60–72 ms | 76–78 ms | 105 ms |
| MuJoCo Warp grouped `wp.Bvh` refit today | 0.04 ms | 0.05 ms | 0.06 ms |

- **One IAS per world doesn't work.** Each `optixAccelBuild` call costs 15–23 µs of mostly host
  overhead, OptiX 9.1 has no batched IAS build, and CUDA-graph capture of many builds fails
  (CUDA errors 900/905/907).
- **A two-level hierarchy (top IAS over per-world IASes) doesn't help.** OptiX cannot start
  traversal at a subtree, which is what MuJoCo Warp's `group_root` does in software.
- **Recommended:** one IAS over all worlds, rebuilt or refitted every step from transforms written
  by a Warp kernel; one GAS per mesh or height field, shared by all worlds; `instanceId` =
  `world · ngeom + geom`.
- **The catch with one IAS:** worlds must be separated spatially, since visibility masks have only
  8 bits. Every ray, shadow rays included, must have its tmax clipped to its world's cell.
  Traversal is 5–10% slower than per-world IASes, and precision drops with large offsets (7e-4 of
  pixels off by more than 1 mm at offsets up to 1.3 km). Using a centred 3D grid of cells should
  reduce this (unverified).

## 3. Traversal throughput on GH200 (software vs software; not RTX)

16.8M primary rays (4096 worlds × 64²):

| per world | Warp `mesh_query_ray` | OptiX bare GAS | MuJoCo Warp-style query | OptiX IAS |
| --- | --- | --- | --- | --- |
| 1 helmet | 6.3 ms | 6.8 ms | 8.9 ms | 7.4–8.1 ms |
| 10 helmets | – | – | 48.3 ms | 33–35 ms |

Without RT cores OptiX is no faster on a bare mesh. With instancing its two-level traversal beats
the MuJoCo Warp-style query by 1.2–1.45×. Hardware gains on RTX remain unmeasured.

## 3b. Measured on RT cores (RTX PRO 6000 Blackwell, Hugging Face Jobs)

Driver 580.178 (OptiX 9.0): PyOptiX must be built against the optix-dev v9.0.0 headers, and the
Warp PTX compiled for an older target (`OPTIX_PTX_ARCH=89`). Warp's CUDA 13 NVRTC PTX for sm_120
is rejected with "Target architecture invalid". The proof of concept passes unchanged: 100% hit
agreement, no validation messages.

16.8M primary rays (4096 worlds × 64², DamagedHelmet), median of 7:

| per world | Warp `mesh_query_ray` | OptiX bare GAS | MuJoCo Warp-style query | OptiX per-world IAS | OptiX single IAS |
| --- | --- | --- | --- | --- | --- |
| 1 helmet | 4.10 ms (4.1 Grays/s) | **0.77 ms (21.9)** | 9.43 ms (1.8) | 0.82 ms (20.5) | **0.85 ms (19.8)** |
| 2 helmets | – | – | 16.7 ms (1.0) | 0.91 ms (18.5) | 0.95 ms (17.7) |

The hardware gives 5.3× on a bare mesh and 11–18× over the two-level query the renderer uses
today. Hit masks agree on 99.995–100% of pixels.

## 4. What an RT backend would change

- **`trace.py`:** a second `make_trace` with the same `trace_closest` / `trace_any` signatures, built
  on trace builtins.
  - The `Hit` is rebuilt from payload registers: t, instance ID → geom/mesh, primitive → face,
    barycentrics.
  - Shadow rays use terminate-on-first-hit with closest-hit disabled.
- **Geometry:**
  - meshes and height fields: shared triangle GASes;
  - box, capsule, cylinder, ellipsoid, plane and SDF: custom-primitive GASes whose intersection
    programs are Warp kernels reusing MuJoCo Warp's `ray_*` functions;
  - flex: per-world GAS updates, or software traversal inside the raygen.
- **`render.py` / `pathtrace.py`:** the megakernel becomes the raygen.
  - `geo, mats, lit, cams, env, out` merge into one params struct (about 5 KB, under the 64 KB
    constant-memory limit).
  - `wp.tid()` becomes the launch index; `RenderOrder` and `block_dim` stop applying.
  - `wp.launch` becomes `optixLaunch` with a cached pipeline per specialised kernel.
- **Per step:** an instance-transform kernel plus an IAS refit, replacing `mjw.refit_bvh`.

## 5. Open risks

- No RT-core hardware was available to measure on.
- The full megakernel as a raygen is untested: OptiX compile time, register pressure, stack size,
  and many trace call sites.
- The single-IAS cell clipping and precision issues above.
- The experimental Warp API: process-global registration, not reload-safe.
- Launch dimensions are limited to about 2^30 rays (unverified).

## 6. Integrated backend (`traversal="optix"`)

Date: 2026-10-22. Code: `mujoco_warp_pbr/_src/optix/`. Option:
`create_render_context(..., traversal="optix")`. RTX PRO 6000 results:
[`../../results/rt_backend.md`](../../results/rt_backend.md).

**Design decisions:**
- **One body, two entry points.** `render.make_render_pixel` holds the former megakernel body as a
  `wp.func`. The software kernel calls it after `wp.tid()`. The OptiX raygen calls it after the
  launch index, with the OptiX `trace_closest` / `trace_any` (`optix/trace.py`).
  - The software path is otherwise unchanged and stays the default.
  - Its RTX PRO 6000 timings are unchanged: 12.9–13.1 ms for damaged_helmet at 4096 × 64², and a
    score of 426k.
- **Modules.** Every build defines its raygen, miss, two closest-hit and one intersection program
  per primitive type in a fresh, uniquely named Warp module (`mujoco_warp_pbr_optix_N`).
  - The module is compiled to PTX with `wp.compile_aot_module`, which caches it by content hash.
  - Entry names are read back from the PTX.
  - If OptiX rejects the device's PTX target, the build retries with sm_89. Driver 580 / OptiX 9.0
    needs this on sm_120. `MJPBR_OPTIX_PTX_ARCH` forces a target.
- **Acceleration structures** (layout of 2026-10-23):
  - one triangle GAS per MuJoCo mesh, built from MuJoCo Warp's `wp.Mesh` points and indices, in
    the authored frame R where `mesh_bvh_frame="auto"` re-frames the software BVH (instance
    transform `geom_xmat @ R^T`; the normal still comes from the original mesh);
  - boxes: instances of one unit-cube triangle GAS scaled to the box (hardware triangles; the
    normal is the hit cube face's);
  - spheres and capsules: one world-space GAS of OptiX built-in round linear curves (linear swept
    spheres, hardware on Blackwell), a capsule one segment along its axis, a sphere a zero-length
    segment (PyOptiX 9.0 has no sphere build input). The closest-hit program re-runs MuJoCo Warp's
    `ray_sphere` / `ray_capsule` on the hit geom, so distances and normals match software;
  - ellipsoids and cylinders: one world-space custom-primitive GAS whose intersection programs
    run MuJoCo Warp's `ray_*` on the translated world pose;
  - one IAS over all worlds' mesh and box instances plus one identity instance per world-space
    GAS. SBT: triangles, curves, then one record per custom type.
- **Planes** are not instanced: being infinite or larger than the scene, they would inflate the
  world cells. `trace_closest` / `trace_any` intersect them analytically with `ray_plane` and
  apply MuJoCo Warp's tie-break.
- **World cells.**
  - Each world is translated onto a cell of a centred 3D grid.
  - Each cell is the union of all worlds' instance bounds. The bounds only grow, and are updated
    on the device every frame with atomics, with no host synchronisation.
  - Every ray, shadow rays included, is clipped to its cell, so a ray starting outside its cell
    (e.g. the camera) begins at the cell entry.
- **Per frame:** small Warp kernels write the bounds, cell, curve segments, custom AABBs,
  triangle instances and the launch parameters (`Scene.write`, `Pipeline.store`). They are
  followed by the OptiX calls: refits of the world-space GASes and the IAS
  (`optix_rebuild_every` rebuilds instead) and `optixLaunch` (`Scene.build`, `Pipeline.trace`).
- **Hit reconstruction.**
  - Mesh hits carry the instance id (world · ngeom + geom), the primitive index (face) and
    barycentrics.
  - The raygen remaps the barycentrics, since OptiX's (b1, b2) weight vertices 1 and 2 and MuJoCo
    Warp's (u, v) weight vertices 0 and 1: u = 1 − b1 − b2 and v = b1.
  - The normal is `geom_xmat @ mesh_eval_face_normal`.
  - Custom primitives pass a world normal and the geom code as attributes.
- **Culling and transparency.**
  - Closest rays use `CULL_BACK_FACING_TRIANGLES`; MuJoCo Warp's winding agrees.
  - Primitive intersection programs apply MuJoCo Warp's exit-face cull.
  - Shadow rays use terminate-on-first-hit with closest-hit disabled, and do not cull triangles,
    as MuJoCo Warp's any-hit.
  - Geoms with alpha 0 get visibility mask 0 (meshes) or a point AABB plus a visibility flag
    (primitives).
- **Launch parameters.** A Warp kernel stores the `Params` struct (all render structs) into the
  device buffer that `optixLaunch` reads. Sub-structs are re-assigned only when their bytes
  change, since Warp deep-copies on struct assignment. This brought the host cost from 0.7 ms to
  0.3 ms per frame.
- **Supersampling.** For spp > 1 the program renders the n × n samples as an n×-resolution
  1-spp image, writing linear HDR. A Warp kernel then tone maps each sample, averages and encodes,
  as the megakernel's sample loop does.
  - Trace calls inside the sample loop made OptiX spill most of the program state across each
    trace: 4.6 KB of continuation spills, 17× slower.
  - The ray-cone footprint now comes from the sub-pixel neighbour, which changes texture LOD
    marginally (≤ 1 level of 8-bit RGB difference).
- **Register cap.** 64 registers per thread (`optix_max_registers`) is fastest; OptiX's default is
  up to 20% slower. Scenes with mirror reflections (a second trace and shading site) spill at 64:
  the default is 96 for them (primitives scene 5.54 → 3.88 ms at 4096 × 64²).
- **`traversal="auto"`** picks OptiX when meshes are rendered and it is available, else software.

**Not supported or different:**
- **CUDA graphs.** Stream capture records the parameter memcpy of `optixLaunch` but not its
  kernel, on OptiX 9.0 (RTX) and 9.1 (GH200), whether captured through Warp or through
  `cuStreamBeginCapture` in any mode. Nsight Systems shows why: the ray-tracing kernel is
  submitted by the driver outside the CUDA API (attributed to the parameter memcpy) and runs
  immediately even during capture, reading a parameter slot the captured copy never filled.
  Captured accel builds replay but break OptiX's next eager build.
  - Opt-in `optix_graph_capture` (`optix/graph.py`): the graph holds the frame's Warp kernels and
    two libc host nodes (`sem_post`, `sem_wait`) around the OptiX calls, which a worker thread runs
    on its own non-blocking stream. Replays match eager renders exactly.
  - Caveat: a pageable device-to-host copy (`array.numpy()`) issued while such a graph runs
    deadlocks (it holds a driver lock the worker's launches need while waiting for the graph);
    synchronise first. GPU-side gating of pre-enqueued work (`cuStreamWaitValue32`) avoids that
    but makes `wp.synchronize()` wait forever on the pending work.
  - Single world 256²: 3.2–3.6k → 3.7–6.7k frames/s; batches gain nothing (0–14% slower), so
    bench.py still times OptiX eagerly.
- **Fallback to software** (with a warning) for flex, height fields, SDF geoms, the path-traced
  tier, a missing PyOptiX or missing OptiX headers.
- **Exact ties** between overlapping surfaces may resolve differently from the software path
  (higher geom index wins there).
- **Primitive-only scenes.** With hardware curves and cube instances, material_grid (spheres) is
  1.26–1.36× faster than software. The primitives scene stays 0.80–0.92× of software, and the
  cause is its mirror floor, not its primitives (without reflections: 2.37 vs 2.41 ms).
  - Tracing the mirror ray through the camera-layer loop (one trace and shading site) was exact
    in the software kernel but wrong in the OptiX program (rows of floor pixels), so it was
    dropped.
  - The first version's instance-per-primitive layout (4.85 vs 5.80 ms on primitives) was
    superseded by the hardware curve / cube layout.
  - MuJoCo Warp's BVH inside the program was dropped. Warp's `bvh_query` uses a shared-memory
    stack, which OptiX rejects, and the ordered traversal there was 1.3–2.6× slower.
