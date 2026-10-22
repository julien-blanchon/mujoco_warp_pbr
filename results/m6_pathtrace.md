# M6 — path-traced tier (2026-10-22)

`tier=Tier.PATH`: unidirectional path tracing with next-event estimation of every light
(bulb-disc sampling for soft lights), environment importance sampling (luminance × sin θ CDF)
combined with BSDF sampling (GGX visible normals + cosine diffuse) by the power heuristic,
MuJoCo reflectance as a stochastic mirror lobe, stochastic alpha, Russian roulette after
`rr_depth`. Same BRDF as the real-time tier; deterministic counter-based random numbers.

## Quality vs Mitsuba 3 (`benchmarks/validate.py --tier path`, 256², 1024 spp both)

Mitsuba renders with a plugin of our BRDF, so both sides integrate the same material model.

| Scene | depth 1 vs `direct`: ꟻLIP / PSNR | depth 4 vs `path`: ꟻLIP / PSNR |
| --- | --- | --- |
| grid_point | 0.001 / 61.7 dB | 0.009 / 37.7 dB |
| grid_sun | 0.002 / 58.8 dB | 0.012 / 36.6 dB |
| furnace | 0.004 / 49.8 dB | 0.003 / 54.4 dB |
| water_bottle | 0.015 / 45.9 dB | 0.015 / 45.8 dB |
| grid_env | 0.016 / 40.8 dB | 0.019 / 38.1 dB |
| damaged_helmet | 0.023 / 38.1 dB | 0.024 / 37.2 dB |
| normal_tangent | 0.036 / 28.6 dB | 0.044 / 27.5 dB |

Residuals are Monte-Carlo noise of both renderers, the 2 cm default MuJoCo bulb radius (soft point
lights for us, true points for Mitsuba) and, for normal_tangent, Mitsuba's own normal-map frame.
Compared with the real-time tier (`results/m5_baseline.md`), environment visibility and
interreflections remove most of the IBL-scene error (grid_env 0.116 → 0.016).

Two issues found and fixed while validating: the segment after the last bounce must still be
traced (it carries the BSDF half of the environment MIS), and the path tracer must not apply
footprint-based specular AA / texture LOD (samples resolve both).

## Speed (1024 worlds, 64², studio HDRI, max_depth 4)

| Scene | 1 spp | 16 spp |
| --- | --- | --- |
| material_grid | 12.1 ms (84k frames/s) | 217 ms (4.7k frames/s) |
| damaged_helmet | 41.2 ms (25k frames/s) | 708 ms (1.4k frames/s) |

≈ 95 M paths/s on DamagedHelmet. The kernel hits 255 registers with local-memory spills; wavefront
or hybrid scheduling is an M7 candidate for this tier.
