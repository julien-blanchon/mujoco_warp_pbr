# NVIDIA Warp: performance features relevant to a batched ray tracer

Survey date: 2026-10-22. Web sources plus the local Warp checkout
`context/warp` (HEAD f5ef098, 2026-10-22;
CHANGELOG top entry 1.18.0, 2026-10-05). Facts marked "(local src)" were read from that checkout.

---

## 1. Introducing Tile-Based Programming in Warp 1.5.0 (NVIDIA developer blog)

- URL: https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0 (updated for the 1.6 API)
- Tiles are 2D, compile-time-shaped arrays that live across a thread block, in registers or shared memory. `wp.tile_load` and `wp.tile_store` are cooperative. `wp.launch_tiled(kernel, dim=grid, block_dim=N)`.
- `tile_matmul` uses cuBLASDx (Tensor Core MMA) and `tile_fft` uses cuFFTDx. GEMM reaches about 70-80% of cuBLAS for large FP32 on A100. Claims up to 4x over tensor frameworks for robot forward dynamics.
- The model is labelled "preview": APIs and performance may change, GEMM outputs go to shared memory, and tile dims must be constants.
- Current docs (https://nvidia.github.io/warp/stable/user_guide/programming_model/tiles.html) describe tiles as cooperative abstractions inside SIMT kernels. Warp also ships tile radix sort, scan, reduce, Cholesky/solve, and **tile BVH/mesh queries** (local src: `warp/native/tile_bvh.h`, `tile_mesh.h`, `tile_radix_sort.h`, `tile_scan.h`).

## 2. Warp BVH / Mesh / Texture primitives (local src + docs)

- `wp.Bvh(lowers, uppers, constructor=..., groups=..., leaf_size=1)` (local src `warp/_src/types.py`):
  - Constructors: `"sah"` (CPU top-down, 16 SAH buckets, best query performance), `"median"`, `"lbvh"` (GPU, default for GPU trees), `"cubql"` (experimental, converted to Warp's layout, no groups).
  - **`groups`** builds one BVH containing many per-group (per-world) subtrees. `wp.bvh_get_group_root()` gives a group-restricted traversal, and `wp.mesh_get_group_root()` exists too. This is a ready-made multi-world TLAS-like structure. A 1.17-era fix stopped group roots leaking into later groups (CHANGELOG line 526).
- Node layout (local src `warp/native/bvh.h`):
  - **Binary BVH stored as two SoA arrays** `node_lowers`/`node_uppers` of `BVHPackedNodeHalf {float x,y,z; uint i:31, b:1}` (16 B each, 32 B per node). Child indices are packed in the `.i` fields, and there are parent/count arrays for refit.
  - Each child test needs separate lower and upper fetches. The node does *not* store both children's boxes together as Aila-Laine's 64 B node does.
  - `BVH_QUERY_STACK_SIZE 32`. `BVH_SHARED_STACK 1` on CUDA (the traversal stack uses shared memory), with `WP_BVH_BLOCK_DIM 256`.
- Queries: `bvh_query_ray`/`bvh_query_next` (iterator returning candidate leaves; the user intersects primitives), `bvh_query_aabb`, `bvh_query_sphere`/`capsule` (1.17), `mesh_query_ray`, `mesh_query_ray_anyhit`.
  - Since about 1.14/1.15, mesh ray traversal is ordered (nearer child first) and reuses loaded payloads, with a faster non-parallel-ray AABB path. It uses more per-thread temporary storage (CHANGELOG line 659).
  - Tile variants `wp.tile_bvh_query_ray`/`tile_bvh_query_next` (the `_tiled` names were deprecated in 1.18) do block-cooperative traversal with a shared stack and result buffer. These target many-results AABB queries more than closest-hit rays.
- Textures: `wp.Texture1D/2D/3D` are CUDA textures with hardware bilinear filtering (MJWarp uses them, needs ≥1.12). **Mipmaps are experimental since about 1.15**: `num_mip_levels`, `mip_filter_mode`, and `wp.texture_sample(..., lod)` (CHANGELOG line 580). This makes ray-cone LOD directly usable with hardware trilinear filtering. Zero-copy interop with external CUDA arrays was added in 1.13. Integer textures cannot be sampled (1.18 breaking change).
- Refit: `wp.Bvh.refit()` and mesh refit support CUDA graph capture, plus CPU graph capture (experimental 1.16).

## 3. Execution and performance knobs (local docs `docs/user_guide/...`)

- Module options default (local src `context.py`): `fast_math=False`, `fuse_fp=True`, `enable_backward=config.enable_backward`, `max_unroll`, `optimization_level`, `cuda_output` (ptx/cubin), `lineinfo`.
  - For the renderer, set `enable_backward=False` (less codegen and compile time; no adjoint), and test `fast_math=True` (approximate rsqrt, div, exp). Newton's camera enables fast math by default.
- `wp.get_cuda_kernel_properties(kernel, block_dim=...)` returns `register_count` and `local_memory_size` (spills). `wp.get_suggested_block_size` gives an occupancy-based block size. Each distinct `block_dim` compiles a separate module variant.
- CUDA graph capture (`wp.ScopedCapture`, `wp.capture_begin/launch`, `wp.CaptureMode`): use it when the same launch sequence repeats and launch overhead matters. Preallocate everything, and keep arrays and BVHs alive while graphs replay. The CUDA 12.6 blog states near-constant repeat-launch CPU cost for straight-line graphs on Ampere+.
- `@wp.func_native` lets us inline CUDA C++ (e.g. PTX `vmin`/`vmax`-style 3-way min/max, warp intrinsics `__ballot_sync`, `__shfl_sync`, explicit `__ldg`/texture fetches) if Python-level Warp codegen is not tight enough.
- CUDA event timing resolution is about 0.5 µs (profiling docs). Use Nsight Compute for register, occupancy, and L1/L2 analysis.

## 4. Existing Warp-based renderers

- **warp.render** (local src `warp/_src/render/render_opengl.py`, `render_usd.py`): an OpenGL viewer with tiled multi-view support (`copy_rgb_frame_tiles` kernels) and a USD exporter. It is a raster debug viewer, not a batch RL renderer.
- **MJWarp batch renderer** (google-deepmind/mujoco_warp PR #1113, 2026-02): a Warp ray caster using Warp BVHs, Lambert, shadows, CUDA textures, and 3DGS. See batch_rendering.md.
- **Newton `SensorCamera`** (previously `SensorTiledCamera`): Warp ray tracer for multi-world rendering. It refits shape BVHs per frame, offers `render_order` PIXEL/VIEW/TILED (tile_width/height), `enable_fast_math` (default on), and `kernel_block_dim`, and produces color, hdr, depth, normal, albedo, and shape_index outputs. Isaac Lab 3.x wraps it as `NewtonWarpRenderer`.
- No published throughput numbers were found for either the MJWarp or Newton renderer beyond "millions of FPS" (MJWarp README; no setup given).

---

## Implications for our renderer (ranked by expected impact)

1. **Write our own BVH node layout and traversal in Warp.** Store both child AABBs in one 64 B parent node (Aila-Laine/tinybvh BVH_GPU), or use a quantized 4-wide or 8-wide node, read as `vec4` loads. Warp's built-in BVH uses split SoA half-nodes, a 32-entry stack, and a general iterator API, which costs extra dependent loads and per-query overhead. Keep `wp.Bvh` (with `groups`) and `mesh_query_ray` as correctness baselines and first-light implementations.
2. **Use `groups` or our own per-world root table for the TLAS.** Try one grouped Warp BVH over all worlds' geoms (refit per step, rebuild on reset) vs a custom LBVH per world. Measure refit+trace vs rebuild+trace.
3. **Use Warp mipmapped textures for ray-cone LOD** (`texture_sample(..., lod)`), with hardware trilinear filtering for baseColor, ORM, and normal. Keep a fallback path for builds without the experimental mip API. Pack small textures into atlases or arrays to limit handle count.
4. **Megakernel hygiene:** `enable_backward=False`, check `register_count` / `local_memory_size` for every kernel variant, keep the traversal stack small (8-16 entries in registers, as with short-stack methods), and sweep `block_dim` (64/128/256). Try `fast_math`. Capture the full step (TLAS update + render + tone map) in a CUDA graph so per-step launch overhead is about 0.
5. **Use `wp.func_native` sparingly for hot inner loops** (ray-box slab with `fminf`/`fmaxf` chains, `__float_as_int` tricks, warp-vote termination) if Warp codegen output (inspect the generated .cu in the kernel cache) is suboptimal.
6. **Tile API is useful for post-processing, not traversal.** Tile-based à-trous/SVGF, SH projection, env-map prefiltering (tile reductions), and tile radix sort for Morton-sorting TLAS instances or ray-sorting experiments. Tile BVH queries are block-cooperative AABB queries, which do not suit per-pixel closest-hit traversal.
7. **Match Newton/MJWarp output conventions** (packed uint32 RGBA, `(nworld, npixel)` buffers, separate hdr, depth, normal, albedo, and segmentation) to drop into both ecosystems.
