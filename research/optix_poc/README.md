# Warp → OptiX proof of concept

Warp kernels compiled as OptiX programs (raygen, miss, closest-hit, any-hit, intersection) through
Warp's experimental external-compilation API, launched with PyOptiX. Findings are in
[`../web/optix_warp.md`](../web/optix_warp.md).

Requirements:
- warp-lang ≥ 1.18;
- PyOptiX (`pip install pyoptix`, which builds from source and needs the CUDA toolkit and CMake);
- an NVIDIA driver with the OptiX runtime;
- the OptiX SDK headers (https://github.com/NVIDIA/optix-dev).

```bash
export OPTIX_INCLUDE_DIR=/path/to/optix-dev/include
python poc_depth.py       # depth image: IAS over a triangle GAS + Warp-intersected spheres, vs Warp
python poc_features.py    # textures and mesh_query_ray inside a raygen
python bench_as.py        # IAS build / refit costs for 4096 worlds
python bench_trace.py     # traversal throughput, OptiX vs Warp
python cap_test.py        # CUDA-graph capture of many accel builds (fails beyond ~64)
```

Files:
- `wp_optix.h`: device glue around `optixTrace` and the payload / ray / hit intrinsics.
- `wpx.py`: builtin registration, build options and PyOptiX pipeline helpers.

Outputs go to `$OPTIX_POC_OUT` (default `./out`). `poc_depth.py` and the benchmarks need the
DamagedHelmet asset from `tools/fetch_assets.py`.
