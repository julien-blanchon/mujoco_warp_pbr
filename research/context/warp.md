# NVIDIA Warp (1.19.0.dev2, last release 1.18.0 2026-10-05)

## Kernel options that matter for a ray-tracing megakernel
- Module options: `fast_math`, `fuse_fp`, `max_unroll` (16), `lineinfo`, `block_dim` (256),
  `optimization_level`; per-kernel `module_options` only with `module="unique"`.
- Per-kernel: `launch_bounds=(maxThreads, minBlocksPerSM)` or `cuda_max_registers=N`
  (CTK ≥ 12.4, mutually exclusive), `enable_cuda_smem_spilling` (CTK ≥ 13), `grid_stride=False`
  (fewer registers), `cluster_dim`. `@wp.func(inline=...)` (1.18).
- Each `block_dim` is a separate compiled variant.
- Inspect: `wp.get_cuda_kernel_properties(kernel, block_dim=)` → `register_count`,
  `local_memory_size`; `wp.get_suggested_block_size`; ncu with `lineinfo`
  (`docs/user_guide/execution_and_performance/profiling.rst`).
- CUDA graphs: `wp.ScopedCapture` + `wp.capture_launch`; `record_cmd=True` launches.

## BVH / Mesh (warp/native/bvh.h, mesh.h)
- `wp.Bvh(lowers, uppers, constructor="sah"|"median"|"lbvh"|"cubql", groups=, leaf_size=1)`;
  default lbvh on GPU. SAH/median built on CPU then copied. `refit()`, `rebuild()` (LBVH on GPU,
  graph-safe).
- `wp.Mesh(..., bvh_constructor=, bvh_leaf_size=4, groups=)`.
- Node layout: **binary**, separate `node_lowers[]`/`node_uppers[]` of 16-byte
  `{float x,y,z; uint i:31, b:1}`, loaded with `__ldg` float4. No wide/compressed nodes.
- Max depth / stack 32 (`BVH_QUERY_STACK_SIZE`); builders force leaves at that depth.
- `mesh_query_ray` / `_anyhit`: local-memory `uint64 stack[32]`, **near-first ordered**, prunes
  by current t, Woop watertight triangle test; tuned for ~62 regs (closest-hit).
- `bvh_query_ray` + `bvh_query_next` (used by mjwarp/Newton for the scene level): stack in
  **shared memory** `__shared__ int stack[32*block_dim]` (32 KB at block 256), **unordered**
  child push, and nested queries in the same block alias the buffer (Newton works around this).
- `tile_bvh_query_ray`: whole block traverses one ray — not useful for per-pixel rays.

## Textures (experimental)
- `wp.Texture2D/3D` = real CUDA texture objects (`tex2D/tex2DLod`); uint8/uint16 normalized,
  float16/32; 1/2/4 channels; CLOSEST/LINEAR; WRAP/CLAMP/MIRROR/BORDER; mipmaps (auto box-filter
  chain, `lod=` explicit — no automatic derivatives). **No cubemaps, no layered, no tex2DGrad.**
- Bindless: `wp.array([...], dtype=wp.Texture2D)` indexed in-kernel (mjwarp uses this).
- Texture fetches return zeros under `llvm_cuda` → use NVRTC.

## Language
- float16/bfloat16 scalars and generic vectors; `wp.static`, closures, function pointers,
  `wp.Function` params, overloads.
- No exposed warp intrinsics (shuffle/ballot) → `@wp.func_native` snippets.
- Tiles: shared-memory tiles, `tile_stack`, `tile_scan/sort/reduce` (useful for compaction in a
  wavefront experiment).
- RNG: PCG (`rand_init(seed, offset)`, `randf`, samplers).
- Interop: zero-copy torch (`wp.to_torch`, `__cuda_array_interface__`), JAX `jax_kernel`.

## Hardware RT
- No RT-core / OptiX support in Warp. Experimental external-compilation API
  (`design/external-compilation-extensions.md`) names an out-of-tree OptiX addon as first consumer.
- GH200 (Hopper) has no RT cores anyway.

## Takeaways
1. Scene-level traversal is the weakest link (unordered + smem stack): our own ordered
   traversal in `@wp.func` is a top optimisation candidate.
2. Tune with `launch_bounds`/`cuda_max_registers`, `grid_stride=False`, block_dim sweep, check
   spills via `get_cuda_kernel_properties`.
3. Env maps as equirect `Texture2D` with mips; LOD from ray cones.
4. Capture render (incl. refit) in a CUDA graph.
