# M8: what RT cores could buy (estimate)

GH200 has no RT cores, so this is an estimate from a measured time breakdown plus published
traversal speedups (`research/web/rt_cores.md`), not a measurement.

## Time breakdown on GH200

Real-time tier, 4096 worlds, 64², 1 spp, HDR environment, one shadow-casting directional light
(one primary and one shadow ray per pixel), camera jitter on, default options (commit 91b71f0).
Times are ms per batch.

| scene | full | no shadows | depth only | primary traversal | shadow traversal | shading and rest | traversal fraction f |
| --- | --- | --- | --- | --- | --- | --- | --- |
| damaged_helmet (mesh, 15k tris) | 23.46 | 13.35 | 8.65 | 8.65 | 10.11 | 4.70 | 0.80 |
| primitives (52 geoms) | 13.88 | 8.81 | 2.02 | 2.02 | 5.07 | 6.79 | 0.51 |
| material_grid (spheres) | 4.46 | 2.99 | 1.42 | 1.42 | 1.47 | 1.57 | 0.65 |

- Primary traversal is the depth-only render, which also counts ray generation and output writes,
  so it slightly overestimates traversal.
- Shadow traversal is full minus no-shadows. Some of that difference is register and occupancy
  cost from the inlined shadow traversal, so f is an upper bound.

## Amdahl estimate

Whole-frame speedup is S = 1 / ((1 − f) + f / k), where k is the traversal speedup from hardware.

| traversal speedup k | helmet | primitives | grid |
| --- | --- | --- | --- |
| 1.5 | 1.36× | 1.20× | 1.28× |
| 2 | 1.67× | 1.34× | 1.48× |
| 4 | 2.50× | 1.62× | 1.95× |
| 10 (NVIDIA marketing upper bound) | 3.57× | 1.85× | 2.41× |
| ∞ | 5.0× | 2.0× | 2.9× |

Which k is realistic:

- Production path tracers gain about 1.4–1.6× end to end from OptiX over CUDA (Blender Cycles),
  and roughly 2–4× on traversal alone.
- Mesh-heavy scenes are where RT cores help most: a plausible gain is about 2–2.5× on DamagedHelmet-like scenes.
- Analytic primitives (boxes, capsules, cylinders, ellipsoids) need custom intersection programs,
  which run on the SMs, so for primitive scenes k stays close to 1–2.
- Spheres (and linear swept spheres on Blackwell) have hardware intersection.

## Caveats

- **Batching.** Thousands of worlds need one TLAS per world or one instanced TLAS over every
  world, refitted every step. Madrona chose software traversal even on RT-core GPUs for this
  reason: "poor throughput under the load of constructing many small top-level BVHs".
  Measure it before assuming k.
- **Different silicon.** RT-core parts (RTX, L40) have other SM counts and memory systems. GH200's
  HBM bandwidth helps the shading and texture side, which this estimate holds fixed.
- **Architecture.** OptiX calls (`optixTrace`) run only inside an OptiX pipeline launch, not
  inside a Warp kernel. An RT backend therefore needs the megakernel as an OptiX raygen program,
  or a wavefront split (Warp generates and shades, OptiX traces), which brings back queue traffic.
- **The seam exists.** All traversal goes through `trace.make_trace` (`trace_closest` /
  `trace_any`), so shading code needs no change. The cost is the launch model, not the shading.

## Recommendation

- Keep the software megakernel as the default; it is the only option on Hopper and Grace-Hopper
  systems.
- An OptiX backend is worth prototyping only for mesh-heavy workloads on RTX or L40 hardware.
  Do it as a wavefront variant that reuses the same shading functions, and first measure per-world
  TLAS refit throughput at 4096 worlds.
