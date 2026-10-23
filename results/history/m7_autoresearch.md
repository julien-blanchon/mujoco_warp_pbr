# M7: auto-research optimisation log

## Setup

- **Score:** geometric-mean frames/s over 13 real-time configurations (`benchmarks/sweep.py
  --preset score`): material grid, primitives and DamagedHelmet at 1024 / 4096 worlds and
  64² / 128², plus one 2×2-supersampled config, with an HDR environment and per-world camera
  jitter (3 cm, 2°).

  Correction: `sweep.py --preset score` never passed `--jitter-camera`, and every row in
  `results/score.jsonl`, the `m7-jitter-*` labels included, records `"jitter_camera": false`.
  The scores below were measured with identical cameras in every world.
- **Hardware:** one GH200. Rows are in `results/score.jsonl` under the labels below.
- **Keep rule:** an idea is kept if the score improves and no scene regresses beyond noise;
  discarded ideas are listed too.
- **GPU sharing:** the GPU was shared with other users' jobs during parts of M7. The busy gate
  in `bench.py` refuses to benchmark while another process uses the GPU, and `sweep.py` retries.
  Runs that raced with other jobs were repeated on an idle GPU.

## Score progression

| step | label | score (frames/s) | vs start |
| --- | --- | --- | --- |
| start of M7 (M5 final, no jitter) | `m5-final` | 113.9k | |
| same code, jittered cameras | `m7-jitter-order0` | 113.8k | 1.00× |
| TILED thread-to-pixel order (8-row bands, warps cover 4 × 8 tiles) | `m7-jitter-order1-b` | 132.2k | 1.16× |
| auto block_dim (256 for primitive-only scenes) + register cap 96 | `m7-regcap96-b` | **184.3k** | **1.62×** |

Per configuration, the last step was 1.36–1.38× on the material grid, 1.7–1.8× on primitives
and 1.08–1.16× on DamagedHelmet (1.41× with 2×2 supersampling).

## Experiments

| idea | result | decision |
| --- | --- | --- |
| `max_unroll` 6 plus `int(wp.static(n))` loop bounds (stop unrolling mjwarp's 8-step mesh query loop) | mesh scenes +25% | kept (J18) |
| `max_unroll=1` module-wide | breaks mjwarp `ray_box` codegen | discarded |
| RenderOrder.TILED | +16% score; faster whether worlds look alike or differ | kept, default |
| RenderOrder.WORLD (neighbouring threads render the same pixel of consecutive worlds) | 2.7× with near-identical worlds (311k score), about 2× slower with strongly jittered cameras | kept as opt-in |
| block_dim 256 for primitive-only scenes (mjwarp default 64) | primitives 1.5×, grid 1.3× | kept, automatic |
| register caps 64 / 80 / 96 / 128 | 64 and 80: helmet −12%, primitives +4%; 96: helmet −11%, primitives −12%, grid −5%; 128 ≈ none | 96 kept, default |
| SAH / LBVH mesh BVH instead of cuBQL | slower | discarded (option kept) |
| depth / segmentation-only cameras skip shading (shading compiled out when no camera renders RGB) | depth only: helmet 22.95 → 8.65 ms, primitives 13.53 → 2.02 ms, grid 4.31 → 1.42 ms (4096 worlds, 64²); faster than mjwarp (21.55 / 2.6 / 1.53 ms) | kept (not in the RGB score) |
| float32 HDR output | correctness fix (float16 overflow at default exposure) | kept |

### Not attempted

- **Analytic primitive UV derivatives.** These would only affect textured primitives (two extra UV
  evaluations per hit), which no score scene exercises heavily.
- **Host-precomputed ray spread.** The in-kernel neighbour-ray cost is one `compute_ray`.
- **func_native traversal snippets / ordered scene-BVH traversal.** These depend on Warp
  internals. The RT-core estimate (`m8_rtcore_estimate.md`) shows traversal at 51–80% of frame
  time, so this is the largest remaining lever.
- **Wavefront path tracing.** Not attempted.

## Lessons

- Warp's kernel cache key does not include `max_unroll`: module-option sweeps need a fresh
  `WARP_CACHE_PATH`, otherwise they benchmark stale binaries (an earlier sweep was invalid for this
  reason).
- On a shared GPU, a single slow configuration is usually contention. Repeat it before
  attributing it to a code change.
