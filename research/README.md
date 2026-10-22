# Research notes

Notes backing the renderer design. Two kinds:

- `context/` — notes on the reference repositories shallow-cloned into `context/` at the repo root
  (gitignored). Paths in these notes are relative to `context/<repo>/`.
- `web/` — notes on papers, blog posts and external reference implementations.

Snapshot date: 2026-10-22. Re-verify version-sensitive facts before relying on them.

| Note | Subject |
| --- | --- |
| [context/mujoco_warp.md](context/mujoco_warp.md) | mjwarp's current ray-cast megakernel renderer (our base) |
| [context/warp.md](context/warp.md) | NVIDIA Warp internals relevant to fast ray-tracing kernels |
| [context/filament.md](context/filament.md) | Filament's PBR model (our real-time shading target) |
| [context/mitsuba3.md](context/mitsuba3.md) | Mitsuba 3 (quality ground truth) |
| [context/newton_isaaclab.md](context/newton_isaaclab.md) | Newton Warp ray tracer, IsaacLab OVRTX/Newton renderers |
| [context/madrona.md](context/madrona.md) | Madrona batch renderer / madrona_mjx |
| [context/mujofil.md](context/mujofil.md) | mujofil (Filament + mjwarp) |
| [context/maniskill.md](context/maniskill.md) | ManiSkill / SAPIEN rendering |
| [context/raytracingpbr.md](context/raytracingpbr.md) | RayTracingPBR (WebGPU MIS path tracer) |
| [web/batch_rendering.md](web/batch_rendering.md) | Batch rendering for embodied AI / RL (Madrona, BPS, Playground, IsaacLab) |
| [web/gpu_traversal.md](web/gpu_traversal.md) | GPU BVH traversal and construction (Aila-Laine, CWBVH, PLOC, wavefront) |
| [web/rt_cores.md](web/rt_cores.md) | RT cores vs software traversal speedups |
| [web/realtime_pbr.md](web/realtime_pbr.md) | Real-time PBR (Filament, split-sum, multiscatter, ray cones, specular AA) |
| [web/denoising.md](web/denoising.md) | À-trous / SVGF denoising and costs |
| [web/warp.md](web/warp.md) | Warp performance notes |
| [web/assets.md](web/assets.md) | Test assets and licenses (Khronos samples, Poly Haven, Menagerie) |
