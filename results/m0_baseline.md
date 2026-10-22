# M0 — setup and baselines (2026-10-22)

Hardware: 1× NVIDIA GH200 480GB (Hopper, sm_90, aarch64), driver CUDA 13.3.
Software: Python 3.12.13, warp-lang 1.18.0 (CUDA 13.4), mujoco 3.17.0.dev, mujoco-warp @ 6c33934,
mitsuba 3.9.1.

## Reference renderers available on this machine

| Renderer | Status |
| --- | --- |
| mujoco_warp renderer | works (baseline below) |
| MuJoCo native `mujoco.Renderer` (EGL) | works |
| Mitsuba 3 `cuda_ad_rgb` | works (Cornell box, 16 spp) |
| ꟻLIP (`flip-evaluator`) | works |
| Newton `SensorCamera` | not installed yet (M5) |

## mujoco_warp render baseline

`benchmarks/run.py` from mujoco_warp, `function=render` (physics excluded). frames/s = nworld · nstep /
run_time, one camera per world; per-call times from the event trace.

| Benchmark | nworld | Resolution | frames/s | render / call | refit / call |
| --- | --- | --- | --- | --- | --- |
| primitives | 8192 | default | 590,018 | 1.67 ms | 0.02 ms |
| unitree_g1_hfield_render (rgb only) | 8192 | 64×64 | 97,971 | 10.19 ms | 0.01 ms |
| cloth_render | 32 | 64×64 | 74,939 | 4.09 ms | 8.69 ms |

Shading in this baseline is MuJoCo's GL look (Lambert + Blinn-Phong, hard shadows), not PBR.

## Assets

`tools/fetch_assets.py`: 19 Khronos glTF sample models (pinned commit) converted to MJCF by
`tools/gltf_to_mjcf.py`, plus 6 Poly Haven 1k HDRIs. All converted models load in MuJoCo and
`mujoco_warp.put_model`.
