# Madrona batch renderer & madrona_mjx

madrona @ 8645708, madrona_mjx @ 1505699 (deprecated → points to MJWarp's renderer).
Won't run on GH200 (custom x86 toolchain/deps submodules, CUDA ≤ 12.5, `jax<0.6`). Idea bank.

## Raster backend (Vulkan, `src/render/batch_renderer.cpp`)
- Per-view frustum cull + indirect draws, 16×16 view atlas per layered target, deferred pass
  untiles. Output effectively unlit albedo (GGX helpers present but unused).

## CUDA ray tracer (`src/mw/device/bvh.cpp`, `bvh_raycast.cpp`, `include/madrona/mesh_bvh.hpp`)
- NVRTC-compiled kernels chained in one CUDA graph: alloc → LBVH build → AABB refit → widen →
  raycast.
- **TLAS per world, rebuilt every frame**: Morton codes (raw float bits — buggy, "FIXME"),
  sort by (world, morton), Karras LBVH, atomic bottom-up refit, collapse to 4-wide.
- **BLAS offline**: Embree SAH, max 2 tris/leaf, spatial splits; de-indexed triangles with
  inline UVs.
- **Node**: `BVHNodeQuantized<uint32_t,4>` ~64 B — float3 origin, int8 per-axis exponent,
  uint8 quantised child min/max, 4 child indices (MSB = leaf). CWBVH-like (Ylitie 2017) at width 4.
  Slab test in quantised space: 1 FMA per plane.
- **Traversal**: one thread per ray, local stack of 64 uint64 "node groups"
  (index | hit-mask | type ∈ {TLAS, BLAS root, BLAS, triangles}); `__ffs(mask)` child pop;
  no front-to-back ordering (TODO); BLAS entry transforms ray, root group pushed to restore.
  Woop watertight triangles with per-instance shear constants. Shadow rays = closest-hit
  (no any-hit early exit).
- **Mapping**: block = 8×8 pixel tile (2 warps), persistent `blockIdx.x` loop over views
  (`gridDim.x = numSMs·4`). Shared-memory stack/prefetch path disabled (`#if 0`).
  Square images only.
- Shading: `max(0.2, Σ clamp(N·L))·albedo`, directional/spot lights, hard shadows, face normals.

## madrona_mjx
- `BatchRenderer(m, gpu_id, num_worlds, view_w, view_h, ...)`; legacy XLA custom calls.
- Per-frame inputs: geom_xpos, geom quats, cam pose; per-world DR: geom_matid, geom_rgba,
  geom_size, lights. Primitives are tessellated; no ellipsoid/hfield.
- No measured numbers in repo ("hundreds of thousands" FPS claim); see paper notes in `web/`.

## Ideas to port (ranked)
1. Per-world TLAS + static SAH BLAS; for small geom counts consider brute-force instance AABB
   tests instead of a TLAS.
2. Quantised 4/8-wide nodes, one 128-bit aligned load per node.
3. Node-group bitmask stack.
4. Do what Madrona left as TODO: front-to-back ordering, any-hit shadows, short smem stack.
5. 8×8 tiles with persistent view loop; keep a warp in one world/camera.
6. De-indexed triangle leaves with inline UVs; precomputed Woop shear per instance.
