# ManiSkill 3 / SAPIEN rendering

- Shader packs (`mani_skill/render/shaders.py`): `minimal` (default for cameras, 8-bit color +
  pos/seg), `default` (float color, normal, albedo), `rt` (32 spp, depth 16, OptiX denoiser),
  `rt-med` (4 spp, depth 3), `rt-fast` (2 spp, depth 1).
- Batched GPU rendering (`mani_skill/envs/scene.py`): physics poses → renderer on GPU
  (`set_cuda_poses` / `GpuSyncManager`), one camera group per sensor, `update_render()` once for
  all envs. **Ray tracing is single-env only** (`sapien_env.py:252`).
- Materials: SAPIEN `RenderMaterial(base_color, metallic, roughness, specular)`; GLB/OBJ PBR
  textures via loaders.
- Lighting: ambient 0.3 + 2 directional lights; env maps per sub-scene (`.hdr`).
- Benchmarks (`mani_skill/examples/benchmarking/gpu_sim.py`, `scripts/maniskill.sh`): FPS =
  env-steps/s including PhysX sim; sweeps over num_envs, cameras, resolution.
  Reported RTX 4090, `minimal`: Cartpole 1×128² rgb 26.8k (n=128) / 32.1k (n=1024);
  1×256² 24.2k; 1×512² 12.3k; PickCube 1×128² 17.5k (n=1024).
- Platform: Vulkan required; SAPIEN aarch64 wheels likely unavailable (unverified).

## Takeaways
- Comparison only meaningful against `default` shader pack (PBR raster), on x86 RTX (Tier 3).
- Their sweep design (cameras × resolution × num_envs, real-dataset camera setups like
  DROID 3×320×180, Open-X 640×480) is a good template for our sweep axes.
