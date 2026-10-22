# mujofil (tau-intelligence/mujofil 0.2.6)

MuJoCo Warp physics + forked Filament 1.56.3 rasteriser; zero-copy torch uint8 `(N,H,W,4)`.

- Native core `native/vendor/core/scene_bridge.cpp` reads `mjModel` (offsets from vendored
  mujoco 3.9 headers ⇒ pins `mujoco>=3.9,<3.10`). Per step only geom poses go to host.
- Backends: surfaceless EGL GL (default, GL→CUDA interop) and experimental Vulkan.
  Layered patch: `gl_Layer = instance_index`, N worlds in one instanced draw (≤256).
- Material mapping (`resolve_material`, `scene_bridge.cpp:213-335`): native metallic/roughness
  when authored; roughness fallback `1 − shininess` clamped [0.04,1]; reflectance ← `mat_specular`;
  **heuristic defaults for unauthored materials** (white → satin plastic, dark neutral → glossy
  black, mid grey → metallic 0.6, saturated → glossy plastic); planes forced matte;
  `mat_reflectance` ignored. MJCF materials: only base-colour texture (normal/MR maps only for
  GLB environments).
- Lights: diffuse magnitude → lux/lm (×13000 directional, ×60000 point, ×150000 spot); spot
  inner cone 0.6× outer; static poses; default sun+fill if none.
- IBL: KTX cubemap + SH, bundled studio HDR, hidden skybox by default.
- Presets: high/eval, medium, fast/train, ultra, raw (SSAO, shadows, MSAA, bloom, FXAA,
  tone mapping toggles).
- Reported (RTX 4060 Laptop): 128px N=512 10.7k cam/s vs vanilla mujoco 8.4k; warehouse with
  IBL + 16 spots + SSAO ~3.2k cam/s; mjwarp raycaster ~42k cam/s at N=2048 (Lambert).
- **aarch64 unsupported** (hardcoded x86_64 paths, x86 prebuilt Filament, x86 wheels only).

## Takeaways
- Its light-unit calibration constants and unauthored-material heuristics are a useful
  starting point for our "physical" fallback mapping (D5) — but note our decision D3 is
  native fields only; heuristics would only apply when nothing is authored (to discuss).
- Speed peer only on an RTX/x86 box (Tier 3).
