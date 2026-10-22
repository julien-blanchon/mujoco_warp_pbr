# RT cores vs software traversal, and what Hopper gives us instead

Survey date: 2026-10-22. Local check: `nvidia-smi` reports "NVIDIA GH200 480GB" with 97,871 MiB
(the 96 GB HBM3 variant).

---

## 1. NVIDIA Turing launch claims (2018)

- Source: NVIDIA Turing architecture launch and whitepaper (not read directly). Secondary coverage: AnandTech https://www.anandtech.com/show/13346/..., HotHardware, bit-tech.
- "10+ Giga Rays/s" on RTX 2080 Ti (68 RT cores) vs about 1.1 Giga Rays/s on GTX 1080 Ti doing software traversal on shader cores. That is a "~10x" claim, reported as a geometric mean over NVIDIA-chosen ray-tracing workloads, not a peak (GameStar quotes 1.21 GR/s for the 1080 Ti).
- This is marketing: different GPU generations, unknown scenes and ray mix. Treat the 10x as an upper bound for incoherent rays, not a same-silicon traversal speedup (unverified detail).

## 2. Measured same-GPU comparisons (sparse literature)

- **Blender Cycles OptiX vs CUDA, RTX 4090** (Renderjuice summary of Blender Open Data medians, secondary, unverified):
  - Junkshop 6.0 s vs 8.7 s (1.45x), Monster 3.1 s vs 4.4 s (1.42x), Classroom 7.4 s vs 11.8 s (1.59x).
  - These are end-to-end path-tracing renders, and shading is shared. The **whole-renderer gain from RT cores is about 1.4-1.6x** on a mature production renderer.
- **WGSL compute-shader BVH vs wgpu ray query, NVIDIA GB10** (GitHub issue tangency-labs/solidcraft#3236, Oct 2026, single anecdote):
  - About 4x end-to-end at 1080p, 64 spp (8.80 s to 2.18 s). About 1.08x at 1 spp, which is dominated by fixed costs.
  - A naive software BVH loses more against RT hardware than an optimized one does.
- **Non-graphics RT-core papers:**
  - RTPD (RTX 3080): about 4.1x average, 6.4x peak over a CUDA implementation for penetration-depth queries.
  - RTCUDB: hundreds of times over OptiX's own software traversal path on Pascal, but that baseline is not optimized.
  - RT-based graph/set methods sometimes lose to CUDA because of fixed RT pipeline overhead.
- **Ray reordering with RT cores** (Meister et al. I3D 2020): reordering gives 1.3-2.0x trace speedup on a 2080 Ti, but the sort overhead is hard to recover when traversal is already hardware-fast.
- **Madrona** (SIGGRAPH Asia 2024): chose a software CUDA tracer even where RT cores exist, because OptiX/Vulkan RT "demonstrate poor throughput under the load of constructing many small top-level BVHs". Its H100 software RT beat the H100 rasterizer by up to 150x (see batch_rendering.md).
- **Intel HPG 2020 keynote** (Holger Gruen, "Are we done with hardware ray tracing?", https://highperformancegraphics.org/slides20/monday_gruen.pdf):
  - On a 2080 Ti, 225 skinned characters cost about 6 ms for animation plus BVH refit, vs 1.8 ms to rasterize.
  - A dynamic BVH costs about 60-80 B/tri, a static one about 30-50 B/tri.
  - Traversal-step divergence: the slowest lane in a 32-wide wave takes many times the average step count. HW RT hides traversal divergence from shaders, but software RT does not.

## 3. Hopper/GH200 resources relevant to software RT

- H100 SXM-class die: 132 SMs, about 67 TFLOPS FP32 non-tensor (TechPowerUp 66.9; Guru3D lists 60, so the exact GH200 clock is unverified), 50 MB L2, up to 228 KB configurable shared memory/L1 per SM, no RT cores, and a small fixed-function raster/geometry allocation (Madrona: H100 raster about 400 FPS on HSSD).
- GH200 GPU memory: 96 GB HBM3 (our node) or 144 GB HBM3e. Bandwidth is about 4 TB/s for HBM3 (NVIDIA marketing figure, not re-verified in this survey) and 4.9 TB/s for HBM3e (secondary). NVLink-C2C runs at 900 GB/s to Grace, with LPDDR5X up to 480 GB at about 500 GB/s.
- Aila et al. 2012 showed software traversal scales with FLOPS. A Hopper SM has 128 FP32 lanes, and H100 FP32 is about 5x an RTX 2080 Ti (13.4 TFLOPS), so a well-tuned software tracer on GH200 should land in the same order as Turing-class hardware RT for coherent rays (inference, unverified).
- The large L2 (50 MB) can hold the BLAS for typical MuJoCo Menagerie scenes (robots of 10^4-10^5 tris at about 8-40 B/tri, i.e. under 5 MB) plus all per-world TLAS. Traversal of small scenes should be L2-resident and therefore ALU-bound.

---

## Implications for our renderer (ranked by expected impact)

1. **Do not plan for RT-core parity tricks.** Optimize for ALU: the slab test, octant ordering, early-out shadow (any-hit) traversal, and analytic primitives. For any-hit shadow rays, terminate on the first hit and skip closest-hit sorting.
2. **Lean on L2 residency.** Keep the BLAS compact (quantized/compressed nodes, indexed tris) so everything fits in the 50 MB L2 across thousands of worlds. Per-world data must be the TLAS and transforms only, never per-world BLAS copies.
3. **Budget estimate:** a mature software tracer on a big FP32 GPU reaches about 0.5-3 Grays/s for coherent primary rays on 10^5-10^6 tris (from CWBVH Pascal numbers of about 1-2.6 Grays/s, scaled; unverified for Hopper). At 128² with 1 primary + 1 shadow ray per pixel (about 33K rays per frame), 2 Grays/s means about 60K frames/s before shading. That is consistent with Madrona's 30K FPS at 128² on HSSD.
4. **Mitigate traversal-step divergence** (Gruen) by mapping pixel tiles to warps, bounding the BLAS depth, and putting simple MuJoCo primitives in TLAS leaves so most rays never enter a triangle BLAS.
5. **Quote RT-core comparisons honestly** in any write-up: 1.4-1.6x whole-renderer (Cycles) to about 4x (naive compute vs ray query), 10x only for marketing. On GH200 the relevant baseline is Madrona btrace on H100, not OptiX.
