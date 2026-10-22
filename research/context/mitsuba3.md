# Mitsuba 3 (3.10.0.dev1, no submodules checked out)

Role for us: **quality ground truth** (both tiers) and path-tracer speed peer.

- BSDFs (`src/bsdfs/`): `principled` (Disney 2015: base_color, roughness, anisotropic,
  metallic, spec_trans, eta/specular, spec_tint, sheen, clearcoat...; `principledhelpers.h`),
  `roughconductor`, `roughdielectric`, `roughplastic`, `diffuse`, `twosided`, `blendbsdf`,
  `mask` (alpha), `normalmap`, `bumpmap`. Tests in `src/bsdfs/tests/`;
  `src/python/python/chi2.py` = χ² test of sampling vs pdf (reusable for our BSDF sampling).
- Emitters: `envmap` (Hierarchical2D importance sampling), point, spot, directional, area,
  constant.
- Integrators: `path` (NEE + MIS, `max_depth`, `rr_depth`), `direct`, `aov` (albedo, depth,
  sh_normal, uv, ...), `ptracer`, `volpath`.
- Variants: `cuda_ad_rgb` (OptiX, dynamic CUDA load), `llvm_ad_rgb`, `scalar_rgb`.
- Batch: `src/sensors/batch.cpp` renders several sensors into one tiled film.
- Scene format: XML or `mi.load_dict`. Test scenes live in the `resources/data` submodule (not
  cloned).
- **aarch64 wheels since 3.8.0** (`docs/release_notes.rst`); `cuda_ad_rgb` on GH200 still to
  verify (`mi.set_variant('cuda_ad_rgb')`).

## Matching our model in Mitsuba
- glTF metallic-roughness ≠ Disney principled exactly. Options: `principled` with
  clearcoat=sheen=spec_trans=0, specular=0.5 (≈ F0 0.04), roughness = perceptual roughness —
  then verify Mitsuba's α mapping (`calc_dist_params`) matches α = pr²; or a `blendbsdf` of
  `roughconductor` (F0 = base color via `specular_reflectance`) and `roughplastic`.
  To decide during validation; differences in diffuse/specular coupling must be quantified.
- Real-time tier comparison: `direct` integrator (+ env map) ≈ our primary + shadow + IBL,
  except split-sum approximations and no env visibility in our IBL.
- Disable Russian roulette (`rr_depth > max_depth`) for deterministic-depth comparisons.
