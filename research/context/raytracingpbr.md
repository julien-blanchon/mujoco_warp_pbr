# RayTracingPBR (HK-SHAO/RayTracingPBR)

WebGPU (TypeScript + WGSL) MIS path tracer; shading in `src/render/shaders.ts`.

- Material: albedo, roughness, metallic, transmission, ior, emission (`struct Bsdf`).
  Lobes: "plastic" (GGX spec with F0 from ior + Lambert × (1−F)), conductor (Schlick F0 =
  albedo, separable Smith G1, VNDF pdf), rough glass (Walter 2007). Mixed by metallic, then
  transmission; delta paths for smooth mirror/glass. α = roughness², min 0.002.
- Sampling: VNDF (Heitz 2018), cosine hemisphere, stochastic lobe selection (`spec_prob` from
  Schlick luminance clamped 0.05–0.9).
- NEE: alias table over lights weighted by lum(Le)·area (`src/render/alias.ts`); sphere cone
  sampling; env-map alias table; MIS power heuristic for NEE and BSDF hits (incl. env).
- Russian roulette from depth 4 (`roulette.ts`); small online path guiding (512-cell hashed
  grid × 8×4 histogram, `guiding.ts`).
- Speed: progressive accumulation, frame-budget pacing, 8×8 workgroups, PCG RNG, branchless ONB
  (Duff et al. 2017), leaf-4 BVH with any-hit early-out, analytic primitives scanned linearly,
  emitter self-skip instead of epsilon offsets.

## Takeaways
- Alias tables (O(1) sampling) for env maps and lights in our PT tier.
- Branchless ONB, self-skip (avoid epsilon offset artefacts) are cheap wins.
- Path guiding is out of scope for v1.
