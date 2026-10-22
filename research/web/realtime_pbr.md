# Real-time PBR shading references

Survey date: 2026-10-22. Tags: "verified" = read in the source text during this survey.
"Prior knowledge" = standard, widely reproduced formulas that I could not re-read this session
(the PDF text was not extractable). Check these against the source before relying on exact constants.

---

## 1. Filament: Physically Based Rendering in Filament (Romain Guy, Mathias Agopian, Google)

- URL: https://google.github.io/filament/Filament.md.html (living document; verified sections below)
- Specular = D·V·F:
  - `D_GGX = α² / (π((n·h)²(α²−1)+1)²)`. In fp16, compute `1−(n·h)²` as `|n×h|²` and clamp to 65504.
  - Height-correlated Smith: `V = 0.5 / (NoL·sqrt(NoV²(1−α²)+α²) + NoV·sqrt(NoL²(1−α²)+α²))`.
  - Fast approximation `0.5/(NoL(NoV(1−α)+α) + NoV(NoL(1−α)+α))`, labeled "mathematically wrong" but acceptable on mobile.
  - `F_Schlick = f0 + (f90−f0)(1−VoH)^5`.
- Remap `α = perceptualRoughness²`. Clamp perceptual roughness to ≥0.089 for fp16 or about 0.045 for fp32 (Frostbite's value).
- Diffuse: Lambert `σ/π` by default, with `σ=(1−metallic)·baseColor`. Burley is optional and not energy conserving.
- `f0 = 0.16·reflectance²·(1−metallic) + baseColor·metallic`. Reflectance 0.5 gives 4%. `f90 = 1`, or `clamp(50·f0.g,0,1)` for micro-occlusion.
- **Multiscatter energy compensation** (Lagarde/Golubev form of Kulla-Conty): `energyCompensation = 1 + f0·(1/dfg.y − 1); Fr *= energyCompensation`. It reuses the split-sum DFG LUT, so it is nearly free.
- **Split-sum IBL**: `spec = (f0·DFG1 + f90·DFG2)·LD(n,α)`.
  - LD is the prefiltered env, assuming v=n=r.
  - The LUT must be ≥16-bit (RG16F); 8-bit is insufficient.
  - Multiscatter variant: store `dfg.x = Σ(1−Fc)·V`, `dfg.y = Σ Fc·V` and reconstruct `mix(dfg.x, dfg.y, f0)`. This is incompatible with Karis/Narkowicz analytic fits.
  - Mip select `lod = maxLod·perceptualRoughness`, e.g. 8·r for a 256² cube with 9 mips. Prefilter with GGX importance sampling plus mip-filtered sampling.
- **Diffuse irradiance**: SH with 2 bands (4 coefficients, a 4×4 mat-vec) or 3 bands (9), pre-convolved with the cosine lobe. Clamp `max(E,0)`.
- Specular occlusion (Lagarde): `saturate(pow(NoV+ao, exp2(−16·r−1)) − 1 + ao)`. Horizon fade (Russell): `h = min(1+dot(r,n),1)`, multiply indirect spec by h².
- Exposure: `EV100 = log2(N²/t) − log2(S/100)`, `exposure = 1/(1.2·2^EV100)`. Pre-expose lights (intensity·exposure) to keep fp16 range.
- Also covers clear coat (Kelemen V, f0 0.04), anisotropy (α_t=α(1+a), α_b=α(1−a), floored at 0.001), cloth (Charlie NDF), Krystek color temperature, and punctual light units. The windowed falloff formula was not in the excerpt I read (see Frostbite).

## 2. Real Shading in Unreal Engine 4 (Brian Karis, SIGGRAPH 2013 PBS course)

- URL: https://blog.selfshadow.com/publications/s2013-shading-course/ (notes PDF was over the fetch limit; prior knowledge)
- D = GGX. G = Schlick-Smith with `k = α/2`, and for analytic lights roughness is remapped `(r+1)/2` before squaring to reduce hotness.
- F uses a spherical-Gaussian Schlick approximation `2^((−5.55473·VoH − 6.98316)·VoH)`.
- **Split sum**: `∫L·f·cos ≈ (1/N ΣL)·(1/N Σ f·cos/pdf)`. The first factor is prefiltered per roughness mip assuming n=v=r (which loses the stretched reflections at grazing angles). The second is a 2D LUT over (NoV, roughness) giving scale and bias on F0.
- Importance-sampled GGX with about 1024 samples offline. The "mip-filtered importance sampling" trick reduces needed samples.
- Area lights use a "representative point" with energy renormalization. Falloff: `saturate(1−(d/r)⁴)² / (d²+1)`.
- Mobile analytic DFG fit (Karis 2014 blog "Physically Based Shading on Mobile"): `EnvBRDFApprox`. An LUT is better on GPU if texture bandwidth is free; in Warp the fit avoids a texture fetch.

## 3. Moving Frostbite to Physically Based Rendering 3.0 (Lagarde & de Rousiers, SIGGRAPH 2014)

- URLs: course notes https://www.ea.com/frostbite/news/moving-frostbite-to-pb, slides https://blog.selfshadow.com/publications/s2014-shading-course/frostbite/s2014_pbs_frostbite_slides.pdf (TOC verified; details are prior knowledge)
- Diffuse is renormalized Disney (energy factor `lerp(1, 1/1.51, linearRoughness)`), combined with height-correlated Smith GGX.
- Physical light units (lumens, candela, lux, nits) and camera exposure (EV100).
- **Smooth windowed inverse-square falloff**: `(saturate(1 − (d²/r²)²))² / max(d², ε)`. Filament uses the same.
- Area lights: sphere, disk, tube, and rectangle via MRP (most representative point) or LTC (later). Specular occlusion formula (used by Filament, above).
- Specular dominant direction for IBL: `lerp(N, R, (1−a)·(sqrt(1−a)+a))` to approximate the off-specular peak in split-sum lookups.
- Energy conservation appendix and the Disney parameter mapping.

## 4. Heitz 2014: Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs

- JCGT 3(2):32-91. https://www.jcgt.org/published/0003/02/03/paper.pdf
- The masking function is the normalization of the visible-normal distribution. Smith is exact for uncorrelated microsurfaces.
- **Height-correlated** `G2 = 1/(1+Λ(v)+Λ(l))` is more accurate than separable `G1(v)G1(l)`. For GGX, `Λ(ω) = (−1 + sqrt(1 + α²tan²θ))/2`, which leads to Filament's V. (Formula is prior knowledge; the citation is verified.)

## 5. Heitz 2018: Sampling the GGX Distribution of Visible Normals

- JCGT 7(4). https://jcgt.org/published/0007/04/01/ (paper text verified for the method)
- GGX is the normal distribution of a truncated ellipsoid. Steps:
  1. Stretch V by (α_x, α_y, 1) to the hemisphere configuration.
  2. Sample a point on the projected disk with a radius-preserving warp of one half.
  3. Project onto the hemisphere and unstretch.
- This is exact, with no rejection and a few lines of code.
- PDF: `D_v(N) = G1(V)·max(0,V·N)·D(N)/V.z`. Reflected-direction pdf `= D_v/(4·V·N)`. With VNDF sampling, the estimator weight reduces to G2/G1 (independent of D and F apart from F itself).
- Successors:
  - **Dupuy & Benyoub 2023**, "Sampling Visible GGX Normals with Spherical Caps" (CGF 42; arXiv 2306.05044): sample a spherical cap and take the halfway vector. Simpler code, up to 39% faster (verified via abstract).
  - **Eto & Tokuyoshi 2023**, "Bounded VNDF Sampling for Smith-GGX Reflections" (SIGGRAPH Asia 2023, AMD): unbiased, with lower variance for rough reflections at small overhead (listing only, unverified details).

## 6. Kulla & Conty 2017: Revisiting Physically Based Shading at Imageworks

- SIGGRAPH 2017 PBS course: https://blog.selfshadow.com/publications/s2017-shading-course/ (slides were mostly non-extractable; prior knowledge)
- Multiple-scattering compensation lobe `f_ms = (1−E(μo))(1−E(μi)) / (π(1−E_avg))`, with E(μ, r) in a 2D table (32²) and E_avg(r) in 1D. Fresnel-colored version: `F_avg·E_avg/(1−F_avg(1−E_avg))` scaling.
- Energy-preserving diffuse-specular coupling: per the glTF spec appendix, a small 16³ table of the Fresnel-weighted specular directional albedo is enough (verified there).
- In a real-time renderer the Filament `1 + f0(1/dfg.y − 1)` scale is the cheap approximation. Missing multiscatter makes rough metals up to about 60% too dark at α=1 (prior knowledge).

## 7. glTF 2.0 specification, Appendix B (BRDF implementation)

- URL: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html (verified)
- Reference model:
  - `c_diff = lerp(baseColor, 0, metallic)`, `f0 = lerp(0.04, baseColor, metallic)`, `α = roughness²`.
  - `F = f0 + (1−f0)(1−|V·H|)^5`.
  - `f_diffuse = (1−F)·c_diff/π`, `f_specular = F·D·G/(4|N·V||N·L|)`.
  - D is GGX with χ+. G is the *separable* Smith; the spec notes height-correlated is more accurate.
  - The informative sample "breaks energy conservation"; Kulla-Conty 16³ tables are suggested.
- Schlick is OK for IoR in [1.2, 2.2]. Recommended for metals over full Fresnel in RGB renderers (Hoffman 2019 F82 mentioned).
- Material conventions:
  - Alpha modes: OPAQUE ignores alpha; MASK keeps alpha ≥ alphaCutoff (default 0.5); BLEND uses "over".
  - doubleSided flips back-face normals.
  - Normal map: `rgb·2−1`, with scale applied to XY before normalize.
  - Occlusion affects indirect only: `1 + strength·(ao−1)`.
  - Emissive is sRGB-encoded, with units cd/m² (nits).
  - baseColor and emissive textures are sRGB; metallicRoughness is linear (G=roughness, B=metallic).

## 8. Khronos PBR Neutral tone mapper

- URL: https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md (verified)
- Constants `F90 = 0.04`, `Ks = 0.8 − F90 = 0.76`, `Kd = 0.15`. Steps:
  1. `x=min(R,G,B)`.
  2. `f = x − x²/(4·F90)` if `x ≤ 2·F90`, else `F90`.
  3. `p = max(c−f)`.
  4. If `p ≤ Ks`, output `c−f`.
  5. Else `pn = 1 − (1−Ks)²/(p+1−2Ks)`, `g = 1/(Kd(p−pn)+1)`, output `(c−f)·(pn/p)·g + pn·(1−g)`.
- Goals: base colors reproduce exactly (minus 0.04) for channels in [0.08, 0.8] under white light. No hue shift, C¹ continuous, invertible, Rec.709 in and out.
- Very cheap (about 15 flops) and per-pixel. Good for RL because albedo stays recognizable and sim-to-real color calibration is simpler. ACES and AgX are not compared in the doc.

## 9. Ray cones: texture LOD for ray tracing

- Akenine-Möller, Nilsson, Andersson, Barré-Brisebois, Toth, Karras, "Texture Level of Detail Strategies for Real-Time Ray Tracing", Ray Tracing Gems ch. 20, 2019 (prior knowledge).
- Akenine-Möller, Crassin, Boksansky, Belcour, Panteleev, Wright, "Improved Shader and Texture Level of Detail Using Ray Cones", JCGT 10(1) 2021, https://jcgt.org/published/0010/01/01/ (verified text):
  - The cone carries width w and spread angle γ. `w' = w + γ·t`. The pixel spread angle is about `atan(2·tan(fovY/2)/H)`. Curvature adds β per hit.
  - The 2019 method needed G-buffer ddx/ddy of position and normal. The 2021 paper adds G-buffer-free curvature estimates (per-triangle from vertex normals; "RC unified" ≈ isotropic ray differentials; "RC combo" over-blurs slightly) and an anisotropic footprint that uses hardware anisotropic filtering.
  - Texture LOD: per-triangle base `Δ = 0.5·log2(t_a/p_a)` (UV area vs world area), then `λ = Δ + log2|w| − log2|n·d|`. Per texture: `λ_i = λ + 0.5·log2(w_i·h_i)` (Listing 2).
  - Guidance: overestimate the spread for Whitted-style (avoid aliasing) and underestimate in path tracers (supersampling handles AA).
  - About 10% total frame-time reduction in a GPU path tracer from mip-coherent fetches. Public-domain code.

## 10. Geometric specular antialiasing (Kaplanyan et al. 2016; Tokuyoshi & Kaplanyan 2019/2021)

- Kaplanyan, Hill, Patney, Lefohn, "Filtering Distributions of Normals for Shading Antialiasing", HPG 2016. Tokuyoshi & Kaplanyan, "Improved Geometric Specular Antialiasing", I3D 2019. **Tokuyoshi & Kaplanyan, "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", JCGT 10(2) 2021**: https://www.jcgt.org/published/0010/02/02/ (verified text)
- Filter the NDF by convolving with the pixel-footprint variance of the half-vector or normal, using orthographic projected space instead of slope space to avoid rim artifacts at grazing angles.
- Simple conservative isotropic version (Eq. 10-11): `α² ← α² + min(2σ²·(‖Δn_u‖² + ‖Δn_v‖²), κ)` with `κ = 0.18` clamp. Use `σ² ≤ 0.25` (half-pixel Gaussian; per Unity HDRP comments, secondary).
- Derivative from neighboring pixels: `‖Δn‖ = ‖n − n'‖`.
- Cost on their test: about 0.1 ms (normal-based isotropic) to 0.5 ms (non-axis-aligned) per frame. Fixes only specular-highlight aliasing, not geometric edges.

---

## Implications for our renderer (ranked by expected impact)

1. **Use the Filament / glTF metallic-roughness BRDF as the single uber-material.** GGX D, height-correlated V (exact form; the sqrt is cheap compared to traversal), Schlick F, Lambert diffuse with `(1−F)` or `(1−metallic)` weighting, and `α = r²` with r clamped ≥0.045. One code path means no material divergence in the megakernel.
2. **Split-sum IBL with a 2-channel DFG LUT (RG16F, about 32²-128²) plus multiscatter compensation.** Use `mix(dfg.x, dfg.y, f0)` and `1 + f0(1/dfg.y − 1)`. Prefilter the env cube or octahedral map once per HDRI on the GPU (GGX importance sampling with mip filtering), with LOD = maxLod·r. SH9 (or SH4) for diffuse. This is a fixed per-scene cost, amortized over all worlds and frames. For per-world HDRI randomization, use a small set of shared prefiltered maps indexed per world.
3. **Ray-cone texture LOD at every primary hit (and at the shadow-ray origin when needed).** Store per-triangle `Δ = 0.5·log2(t_a/p_a)` at BLAS build time. Use `λ = Δ + log2(w) − log2|n·d| + 0.5·log2(W·H)`. This kills texture aliasing at 64-128² and improves cache behavior (about 10% faster in NVIDIA's PT). Underestimate the spread when supersampling in-kernel.
4. **Geometric specular AA without screen derivatives.** We have no ddx in a ray tracer. Use the ray-cone footprint times per-triangle normal curvature (vertex-normal differences) to estimate `‖Δn‖²`, then `α² += min(2σ²‖Δn‖², 0.18)`. At 64² most specular highlights are subpixel, so this matters for temporal stability in RL observations.
5. **Tone mapping: Khronos PBR Neutral by default** (keeps albedo-faithful colors for policy learning), with ACES or AgX optional. Output sRGB uint8 packed like MJWarp. Pre-expose lights so we can stay in fp16 where useful.
6. **Path-traced tier sampling.** VNDF sampling (Heitz 2018, or the Dupuy-Benyoub spherical-cap variant, up to 39% cheaper) with weight G2/G1. Optionally bounded VNDF for rough reflections. MIS against env importance sampling and NEE.
7. **Physical units are optional, but keep Frostbite's windowed inverse-square falloff** so MuJoCo point and spot lights have finite range. That can be used to skip shadow rays for out-of-range lights.
8. **Respect glTF conventions** (sRGB vs linear textures, MASK alpha cutoff 0.5 in any-hit, normal-map scale, occlusion only on indirect) so Khronos sample assets render correctly. These are our conformance tests (see assets.md).
