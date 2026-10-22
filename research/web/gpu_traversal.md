# GPU software ray traversal and BVH construction

Survey date: 2026-10-22. Target: GH200 (Hopper H100 die, 132 SMs, no RT cores). Numbers
from PDFs below were extracted from the paper text (verified) unless marked otherwise.

---

## 1. Understanding the Efficiency of Ray Traversal on GPUs

- Authors/year: Timo Aila, Samuli Laine (NVIDIA), HPG 2009
- URL: https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf
- Setup: GTX285, binary SAH BVH (max 8 tris/leaf, split large tris), both child AABBs in one 64 B node fetched together, Woop unit triangle test (48 B/tri), rays assigned to warps in Morton order, nodes via texture cache, 192-thread blocks, 21-25 registers.
- Known methods ran 1.5-2.5x below a simulated instruction-issue upper bound. The gap was hardware work distribution (long-running warps holding slots), not memory bandwidth.
- **Persistent threads** (launch enough threads to fill the GPU once, fetch ray batches via an atomic counter) made packet traversal 1.5-2.2x faster, reaching 10-20% of the bound.
- **while-while vs if-if loop organization.** if-if has better traversal SIMD efficiency and shorter worst-case warps (about 30% faster slowest warps). Smaller leaves (≤4 tris) favor if-if.
- **Speculative traversal** (1-entry postponed-leaf buffer): +5% primary, +10% AO, no gain for diffuse (bandwidth-bound).
- Conference scene, 282K tris, 1024x768: persistent speculative while-while ran primary 142 Mrays/s, AO 135, diffuse 61 (GTX285). Per-ray traversal beats packets by about 2x for diffuse rays.
- Replacing terminated rays and work queues needed warp popc/prefix-sum instructions (`__ballot`/`__popc`), which now exist.

## 2. Kepler and Fermi Addendum (NVIDIA tech report NVR-2012-02)

- Authors/year: Aila, Laine, Karras, HPG 2012 poster
- URL: https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02.pdf
- Traversal speed tracked peak FLOPS almost perfectly across GTX285→480→680 (779→1344→3090 GFLOPS), while bandwidth only went 159→192 GB/s. Ray tracing was "almost entirely limited by the available FLOPS", so instruction-level optimization matters.
- Conference (283K tris) Mrays/s, Tesla/Fermi/Kepler: primary 142/272/433, AO 135/284/518, diffuse 61/126/245. San Miguel (11M) on Kepler: primary 132, diffuse 59.
- **Persistent threads are no longer beneficial on Fermi and Kepler** because the hardware work distributor improved. On Kepler, replacing terminated rays when fewer than 60% of lanes are active gives +10% for incoherent rays.
- Kepler: fetch all node and triangle data through texture cache and avoid dependent fetches even at the cost of overfetch. L1 is only good for stack and ray fetches.
- VMIN/VMAX (fused 3-operand integer min/max for slab tests) gave +10% on Fermi and +5% on Kepler. Ray-box span min/max is the hottest instruction mix.
- Aila-Laine became the baseline for most later GPU traversal work. Its public CUDA code ("Understanding" kernels) is the reference for `fmin_fmax` tricks.

## 3. Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs (CWBVH)

- Authors/year: Henri Ylitie, Tero Karras, Samuli Laine (NVIDIA), HPG 2017
- URL: https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf
- Node format: 8-wide, 80 B per node (10 B per child). Contents:
  - Local grid origin p (3×fp32) and per-axis power-of-two scale stored as 3 exponent bytes.
  - imask byte, base indices for child nodes and triangles, 8 meta bytes.
  - 6×8 quantized 8-bit planes.
  - Compression is 1:3.2 vs uncompressed. 8-bit quantization adds only +3.4% node and triangle tests (7 bits +6.1%, 6 bits +11.1%).
- **Octant-ordered traversal**: children are stored at build time in a per-octant-friendly order, and traversal order is `slot XOR ray_octant`. This avoids sorting hit distances (a sorting network would need 19 compare-swaps for 8 children) at a small cost: 14.7 vs 13.4 node tests per primary ray compared with exact distance order. Random order gives 17.7.
- Compressed traversal stack: each 64-bit entry is a node group (base + hit mask for up to 8 children) or a triangle group (up to 24 tris). Top entry in registers, and the stack lives mostly in shared memory, which removes "practically all" stack traffic.
- **Triangle postponing**: when the fraction of warp lanes doing triangle tests falls below a threshold, push the remaining triangle group back to the stack to keep the SIMD lanes busy. Persistent threads with dynamic ray fetch are retained.
- Build: CPU SBVH binary tree (1 tri/leaf), then SAH-optimal collapse into a wide BVH by dynamic programming. Leaves have ≤3 tris. This is offline: about 6 min for 10M tris (unoptimized).
- Results (Titan X Pascal, geomean speedup vs Aila et al. 2012): primary 1.36x, 1st diffuse bounce 2.36x, 4th bounce 2.70x. Memory is 0.43x of the baseline (about 8 B/tri for nodes). Example: Conference primary/diffuse goes from 1412/432 to 2642/1224 Mrays/s. The stackless (Binder & Keller 2016) row gives 1.24-1.32x.
- **Gains are mainly for incoherent rays.** Primary rays gain much less (1.1-1.9x per scene).

## 4. Wide BVH Traversal with a Short Stack

- Authors/year: Karthik Vaidyanathan, Sven Woop, Carsten Benthin (Intel), HPG 2019
- URLs: https://diglib.eg.org/handle/10.2312/hpg20191190, slides https://highperformancegraphics.org/wp-content/uploads/2019/session2/wide_bvh_traversal_short_stack.pdf
- Extends restart-trail stackless traversal to wide BVHs. With only 5 stack entries (fits in registers or on-chip), traversal steps are close to a full stack.
- Leaf culling once a closer hit is known: up to -14% primitive tests.
- Read only abstract-level summaries; tables not verified.

## 5. Efficient Stackless Hierarchy Traversal on GPUs with Backtracking in Constant Time

- Authors/year: Nikolaus Binder, Alexander Keller (NVIDIA), HPG 2016
- URL: https://research.nvidia.com/publication/2016-06_efficient-stackless-hierarchy-traversal-gpus-backtracking-constant-time
- Constant-time, constant-memory backtracking via perfect hashing of node keys. In the CWBVH table: 1.24x primary, 1.27x 1st bounce, 1.32x 4th bounce vs Aila 2012 (Pascal). Slower than CWBVH for incoherent rays, but needs no stack memory.

## 6. Megakernels Considered Harmful: Wavefront Path Tracing on GPUs

- Authors/year: Samuli Laine, Tero Karras, Timo Aila (NVIDIA), HPG 2013
- URL: https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf
- Design: a pool of 1M paths, 212 B of state per path (212 MB total) in global memory. Stages: a logic kernel (all paths, MIS and RR), a new-path kernel, per-material kernels, and extension and shadow ray-cast kernels. Fixed-size queues are 4 MB each. Going from 1M to 8M paths gives only about +5%.
- Results on Tesla K20 (Mpaths/s, megakernel vs wavefront):

  | Scene | Megakernel | Wavefront | Speedup |
  |---|---|---|---|
  | Carpaint (9.5K tris, one 4-layer material) | 43.0 | 58.4 | +36% |
  | City (879K) | 5.41 | 9.70 | +79% |
  | Conference (283K, 3 heavy materials) | 2.71 | 8.71 | +221% |

- Megakernel thread utilization in Conference was 23%, vs 53% for wavefront. Ray casts were 44-56% of wavefront time.
- Even with one material, wavefront won because ray-cast kernels run at low register counts (better latency hiding and I-cache fit).
- Megakernel trick: regenerate paths when more than half the warp is idle, worth about 15%. Atomic coalescing of queue pushes (warp ballot) gave +40% overall and sped up ray casts by +32% through better coherence.
- Caveat for us: the benefit comes from material divergence and register pressure. With one uber-BRDF (Filament GGX) and short paths at 64-256², the trade-off is different. The fixed per-launch overhead also matters when a frame is about 10-50 µs per world batch.

## 7. LBVH / Karras 2012 / TRBVH (Karras & Aila 2013)

- Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", HPG 2012: a fully parallel binary radix tree over sorted Morton codes, with all internal nodes built in one kernel. Madrona uses it for per-world TLAS.
- Karras & Aila, "Fast Parallel Construction of High-Quality BVHs", HPG 2013 (https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf): LBVH followed by parallel treelet restructuring (7-leaf treelets, optimal via DP), plus optional triangle splitting. It reaches >90% of SBVH trace performance (prior fast GPU builders reach about 50%) and builds at about 40M tris/s on GTX Titan (abstract-level, verified via search snippet).

## 8. PLOC / PLOC++ / H-PLOC

- Meister & Bittner, "Parallel Locally-Ordered Clustering for BVH Construction", IEEE TVCG 2018 (https://dcgi.fel.cvut.cz/projects/ploc/): agglomerative clustering over Morton-sorted clusters within a search radius r. Up to 2x faster builds than prior high-quality GPU builders and up to 17% faster trace.
- Benthin et al., PLOC++ (HPG 2022): 1.9-2.3x faster than PLOC.
- Benthin, Meister, Barczak, Mehalwal, Tsakok, Kensler (AMD), **H-PLOC**, Proc. ACM CGIT 7(3) 2024, DOI 10.1145/3675377, PDF https://gpuopen.com/download/publications/HPLOC.pdf:
  - LBVH-style bottom-up hierarchy where each subtree merge runs PLOC within a warp, all in a single kernel launch.
  - Quality similar to PLOC++/ATRBVH. Full build 1.1-3.6x faster, binary phase 1.6-13x faster. Includes a single-kernel binary-to-n-wide conversion.
  - The paper's figures include "H-PLOC vs LBVH (Instance TLAS)" build overhead vs instance count (figure values not extractable).
  - Valve's RADV driver adopted H-PLOC for TLAS builds (Phoronix, about 1 ms saved per frame; secondary source). tinybvh ships an H-PLOC builder.

## 9. tinybvh and Jacco Bikker's "How to build a BVH" series

- URLs: https://github.com/jbikker/tinybvh (MIT), https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/ (10 parts, 2022)
- Series parts: 1 basics, 2 SAH and ordered traversal, 3 binned building, 4 refit vs rebuild, 5-6 TLAS/BLAS, 7-8 shading/Whitted, 9-10 OpenCL GPU.
  - Part 1: 32 B node (`aabbMin, leftFirst, aabbMax, triCount`, with right = left+1), at most 2N-1 nodes preallocated, in-place index partition.
- tinybvh layouts: BVH (32 B), BVH_GPU (64 B, both child AABBs in parent, Aila-Laine style), BVH4_GPU (quantized 4-wide), BVH8_CWBVH (80 B, Ylitie), CPU BVH4/BVH8 (AVX2/NEON), double precision. Builders: Bonsai (default), binned SAH (NEON on ARM, useful for Grace CPU builds), SBVH, H-PLOC, TLAS over BLAS instances. `bvhopt` reinsertion optimization gives -8% SAH on Sponza.
- README claim (unverified): "up to 5 billion rays per second" in Crytek Sponza on an RTX 5080 laptop GPU with the basic binary BVH and no RT cores. Ray type and resolution are not given, likely coherent primary rays.
- Offers OpenCL, GLSL, DX12, and Vulkan traversal kernels (Aila-Laine and CWBVH). This is a clean reference to port to Warp.

## 10. On Ray Reordering Techniques for Faster GPU Ray Tracing

- Meister, Bokšanský, Guthe, Bittner, I3D 2020 (arXiv 2506.11273)
- Reordering rays by key (origin/direction Morton or hash) speeds up trace by 1.3-2.0x on an RTX 2080 Ti with RTX trace kernels, but the reorder overhead is hard to recover with hardware RT. With slower software traversal, the trade-off is more favorable (our inference).

## 11. "Ygdrasil"

- Not found under that name or "Yggdrasil" (searched). The nearest relevant items are Binder & Keller's stackless traversal and short-stack wide BVH traversal. Treat as unverified or misremembered.

---

## Implications for our renderer (ranked by expected impact)

1. **Use a wide, compressed, or at least 4-wide quantized BLAS for mesh geoms, built offline on Grace.** Use CWBVH-style 8-wide 80 B nodes, or tinybvh's BVH4_GPU or BVH_GPU (64 B, both children per fetch) as a simpler first step. The measured gains are 1.36x primary and 2.4-2.7x incoherent over a good binary BVH, at 0.43x the memory, so the path tracer benefits most. Build with binned SAH or SBVH on the CPU (tinybvh with NEON on Grace), or H-PLOC on the GPU.
2. **Keep the TLAS tiny and rebuilt per step.** Use LBVH or H-PLOC per world over geoms in one launch, with the Morton sort fused into the world sort (Madrona). For ≤64 geoms per world, a brute-force or 1-level "AABB list in shared memory" may beat any tree. Benchmark against Warp's built-in `wp.Bvh` refit.
3. **Plan for being compute-bound, not bandwidth-bound,** especially for primary and shadow rays at low resolution with scenes that fit L2. Hopper has a 50 MB L2 and 228 KB smem per SM, and Aila 2012 found traversal tracks FLOPS. Micro-optimize the ray-AABB slab test: precompute `invD` and `-O*invD` so each plane is one FMA, use `fminf`/`fmaxf` 3-way chains, and use octant precomputation. Use `wp.fast_math`-style intrinsics where available.
4. **Order children by octant, not sorted distance**, for 4- or 8-wide nodes. It is cheap and nearly as good. Keep the top-of-stack in registers and use short stacks (5-8 entries in registers or local) with a restart fallback. Warp has no shared-memory stack primitives in plain kernels, so a short stack in registers is the practical choice (see warp.md).
5. **Megakernel first; wavefront only where divergence dominates.** Our BRDF is a single uber-shader, so material divergence (the main wavefront win) is small for the real-time tier. For the path tracer, split at least the trace loop from shading if register pressure kills occupancy (check with ncu). Use path regeneration in-warp (about +15%) or the persistent-ray-refill heuristic (<60% active lanes) for the PT tier.
6. **Assign rays to threads coherently.** Use Morton or tile order within a camera, so a warp covers an 8x4 pixel tile. Aila used Morton order. Leave ray sorting for PT bounces ≥1, and only if profiling shows divergence. Reordering costs can exceed the gain at low ray counts per world.
7. **Persistent threads are probably not needed** on modern GPUs for primary rays (Aila 2012 found no benefit from Fermi on). Consider them only for the PT tier with ray refill, and verify on GH200.
8. **Leaf size of 1-4 triangles**, with triangle data in the precomputed Woop or "vertex0, edge1, edge2" format (48 B) to cut ALU per test. Watertightness matters less for RL observations.
