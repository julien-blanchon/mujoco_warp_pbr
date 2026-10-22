# Denoising for the path-traced tier

Survey date: 2026-10-22. The Dammertz and SVGF numbers were read from the paper text (verified).

---

## 1. Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering

- Authors/year: Holger Dammertz, Daniel Sewtz, Johannes Hanika, Hendrik P.A. Lensch (Ulm), HPG 2010
- URL: https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.100/institut/Papers/atrousGIfilter.pdf
- Undecimated à-trous: a 5×5 B3-spline kernel `h = [1/16, 1/4, 3/8, 1/4, 1/16]` applied iteratively with step 2^i. Five iterations give about an 80×80 effective footprint, with a constant 25 taps per iteration.
- Edge-stopping weights combine color (noisy ray-traced), normal, and position buffers: `w = w_rt·w_n·w_x`, each `exp(−‖Δ‖²/σ²)`, with σ for color shrinking per iteration.
- Timing (Table 2, GTX 285, ms):

  | Resolution | 5 iterations, 0 weights | 5 iterations, 3 weights | 10 iterations, 3 weights |
  |---|---|---|---|
  | 512² | 1.5 | 5.6 | 11.0 |
  | 1024² | 5.0 | 20.9 | 41.6 |
  | 1080p | 9.6 | 42.6 | 86.0 |

  Cost is linear in pixels × iterations × number of weights.
- At 1080p with 5 iterations and 3 weights that is about 20 ns/pixel on a 2009 GPU, i.e. cheap and bandwidth-bound. It is a pure spatial filter: blurs fine detail, fails with very low spp, and gives no temporal stability.
- Demodulating albedo (filter irradiance and remultiply by texture) is the key trick that keeps texture detail.

## 2. Spatiotemporal Variance-Guided Filtering (SVGF)

- Authors/year: Christoph Schied, Anton Kaplanyan, Chris Wyman, Anjul Patney, Chakravarty R. Alla Chaitanya, John Burgess, Shiqiu Liu, Carsten Dachsbacher, Aaron Lefohn, Marco Salvi, HPG 2017
- URLs: https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced, preprint `svgf_preprint.pdf` on that page. Reference implementation: Falcor SVGFPass.
- Input: 1 spp path-traced direct and indirect, filtered separately, with albedo demodulated. Needs a noise-free G-buffer (depth, object- and world-space normals, mesh ID, motion vectors). The paper used a raster primary pass and OptiX secondary rays.
- Pipeline:
  1. Temporal accumulation of color and first/second luminance moments via motion-vector reprojection. Exponential blend α = 0.2. Consistency tests on depth, normal, and mesh ID.
  2. Variance estimate from temporal moments, falling back to a 7×7 spatial bilateral estimate when history is under 4 frames.
  3. Five à-trous iterations (65×65 footprint) with edge-stopping on depth (using the depth gradient), normals (`max(0,n·n')^128`), and luminance scaled by sqrt of the 3×3 Gaussian-prefiltered variance. Iteration 1 output feeds the color history.
  4. Remodulate, tone map, TAA.
- Cost: about 10 ms at 1920×1080. 4.4 ms average (4.1-5.8 ms) at 1280×720 on a TITAN X (Pascal), roughly scene-independent, with variability from disocclusion handling.
- Quality: about 10x more temporally stable than prior interactive filters and 5-47% better SSIM. Table 1, RMSE vs 4096 spp: e.g. Sponza 0.022 (SVGF) vs 0.043 (EAW).
- Requirements that matter for RL: per-world history buffers (color, moments, depth, normal, mesh id; about 40-60 B/pixel), valid motion vectors (rigid MuJoCo bodies are easy: reproject with the previous body pose), and frame-to-frame continuity. Episode resets invalidate history.

## 3. Successors worth knowing (not read in depth; unverified details)

- A-SVGF (Schied, Peters, Dachsbacher, HPG 2018): adaptive temporal α from gradient estimation, reducing ghosting.
- NVIDIA NRD (ReBLUR/ReLAX) is the production descendant of SVGF/A-SVGF. Intel OIDN and NVIDIA OptiX denoiser are neural, high quality, and far too heavy per pixel for batched RL at thousands of FPS.

---

## Cost estimate for our workload (back-of-envelope, inference)

- Batch of 1024 worlds × 128² = 16.8 Mpix per step.
- SVGF at 4.4 ms per 0.92 Mpix (Pascal) is about 4.8 ns/pix. GH200 has about 5-6x the FP32 and 7x the bandwidth of TITAN X Pascal, which gives about 0.8-1 ns/pix, or **about 13-17 ms per batch**. That is a 4-8x overhead on a 2-4 ms PT render. Too expensive as a default.
- Five-iteration à-trous with 3 weights scaled from GTX285 (20 ns/pix) by about 30-50x hardware gives about 0.5 ns/pix, or about 8 ms per batch. A fused shared-memory implementation (all 5 levels in one launch per 32×32 tile with halo, which is feasible at 64-128²) could cut that several-fold.

---

## Implications for our renderer (ranked by expected impact)

1. **Default PT-tier output should be undenoised or progressively accumulated.** Offer denoising as opt-in. For RL observations the policy can learn through moderate noise, while denoiser artifacts (blur, ghosting) are systematic biases.
2. **If denoising, start with single-frame à-trous using albedo demodulation and G-buffer edge-stopping** (depth, normal, mesh id, which we get from the primary hit for free). Use 3-4 iterations at 64-128² (the footprint covers the image anyway) in one fused kernel per world tile. Warp's `wp.tile` shared-memory ops may help (see warp.md).
3. **Temporal accumulation is cheap and effective for static-camera episodes** (fixed cameras are common in manipulation). Use motion vectors from rigid body poses. Reset history on episode reset, using the per-world done flag.
4. **SVGF only for demo/eval-quality video**, not training throughput.
5. **Prefer variance reduction at the source**: NEE+MIS, env importance sampling, VNDF, low-discrepancy per-pixel sequences (Sobol with Owen scrambling or per-pixel XOR scrambles, as in Laine 2013), and 2-4 spp in-kernel. This is usually cheaper per unit quality than filtering at these resolutions.
