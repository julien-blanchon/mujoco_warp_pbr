# Batched Omniverse RTX (ovrtx) rendering: options and measurements

Date: 2026-10-23.

**Setup.** RTX PRO 6000 Blackwell Server Edition (driver 580.178), ovrtx 0.5.1, ovstage 0.2.1,
Newton 1.7, warp-lang 1.18. Code: [`benchmarks/references/ovrtx_tiled.py`](../../benchmarks/references/ovrtx_tiled.py)
(bench renderer `newton_rtx_tiled`).

## Batching options

The `newton_rtx` reference wraps Newton's `ViewerRTX`. It renders one camera per call and
reaches ~120-170 frames/s at 64²-256². Four ways to batch it:

| option | through ViewerRTX's public API? | result |
| --- | --- | --- |
| (a) N render products (one camera each) in one `Renderer.step` | no (stage edits) | **no gain**: 166 / 178 frames/s at N = 64 / 256 (64²). Each product costs as much as a separate render, and memory grows (15 GB at N = 256). |
| (b) one render product targeting N cameras, resolution = tile atlas (IsaacLab) | no (stage edits + direct `step`) | **works**: up to 74k frames/s (RTPT, 64²) |
| (c) GPU readback (DLPack `map(device=CUDA, sync_stream=…)` plus one Warp tile-split kernel) and `step_async` pipelining | readback: no. `get_frame()` goes GPU→CPU→GPU. | GPU mapping is needed: a CPU copy of the atlas would dominate. `step_async` pipelining gave no gain (51.5k vs 52.1k frames/s, mg 1024 @ 64²) because the work is GPU-bound. |
| (d) `omni:rtx:rendermode = "Minimal"` (simple shading) instead of RealTimePathTracing | `render_settings=` can set it | 3-8× faster, but our scenes render wrong with ViewerRTX's lights (see Limits) |

The public `ViewerRTX` API supports one camera and one product. `render_settings` can change
the render mode, and `async_rendering` overlaps one frame. Batching needs the stage-level
changes below. They are made in a `ViewerRTX` subclass, so the viewer's USD export still
supplies the meshes, materials and "studio" lights.

## What `ovrtx_tiled.py` does (IsaacLab's isaaclab_ov OVRTX layout)

1. It builds a 1-world Newton model (`newton_scene(spec, mjm, 1)`). ViewerRTX authors its USD
   stage. Then the subclass edits the stage before export:
   - **World copies.** Each world-0 shape batch prim is copied N−1 times in place with
     `Sdf.CopySpec` (C++). IsaacLab clones natively instead (`ovstage.Stage.clone`).
     Authoring N worlds through Newton/ViewerUSD is Python-bound: 20 s for 256 worlds of
     material_grid, so about 5 min at 4096.
   - **Scene partitions.** Each world is tagged with `primvars:omni:scenePartition = "env_i"`,
     and each camera `i` with `omni:scenePartition = "env_i"`. The worlds overlap in space and
     camera i sees only world i (plus unpartitioned lights).
   - **Cameras and product.** N `UsdGeom.Camera` prims take the exact pose and fovy of the
     MuJoCo camera "cam". The viewer's RenderProduct `camera` relationship targets all N
     cameras, with `resolution = (cols·res, rows·res)` and `cols = ceil(sqrt(N))`. As in
     IsaacLab, `omni:rtx:rt:ecoMode:enabled = false`, and
     `waitForEvents = [AllLoadingFinished, OnlyOnFirstRequest]`.
2. **`render()`.** One `Renderer.step({product}, ordinal=…)` renders all cameras. Then the
   LdrColor atlas is mapped on CUDA (stream-ordered) and split into `(N, res, res, 3)` uint8 by
   one Warp kernel. Nothing goes through host memory.
3. **`update()` (optional, `--refit` in bench).** It calls the viewer's per-frame scene update:
   world-0 transforms are written to ovstage, then `advance_write_floor().wait()`. The world
   copies are static.

**Partition check.** The world copies were shifted by 0.6 m·i, with each camera shifted the same
way. All tiles stayed identical, with no ghost geometry from the other worlds.

## Numbers (frames/s = N / time of one `render()`, static scene, eager timing)

RealTimePathTracing, 1 spp, no denoiser (the viewer's settings):

| scene | res | N=1 | 64 | 256 | 1024 | 4096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 64² | 163 | 6.5k | 19.3k | 52.1k | 74.2k |
| material_grid | 128² | 120 | 4.8k | 13.2k | 22.2k | 33.1k |
| damaged_helmet | 64² | 175 | 6.5k | 19.3k | 45.0k | 67.8k |
| damaged_helmet | 128² | 119 | 4.8k | 13.2k | 19.5k | 28.0k |
| franka_showroom | 64² | 160 | 6.5k | 18.6k | 49.8k | 70.4k |
| franka_showroom | 128² | 115 | 4.8k | 13.1k | 20.5k | 30.4k |

Minimal mode (`minimal:mode` 3, shadows on):

| scene | res | N=1 | 64 | 256 | 1024 | 4096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 64² | 634 | 38k | 142k | 225k | 449k |
| material_grid | 128² | 616 | 38k | 62k | 146k | 194k |
| damaged_helmet | 64² | 640 | 39k | 145k | 233k | 547k |
| damaged_helmet | 128² | 624 | 38k | 69k | 149k | 246k |
| franka_showroom | 64² | 617 | 38k | 133k | 200k | 215k |
| franka_showroom | 128² | 564 | 37k | 57k | 148k | not run |

The RTPT throughput hardly depends on the scene, and N = 1 to 64 is overhead-bound: one step
costs at least about 6 ms (RTPT) or 1.6 ms (Minimal).

Comparison, suite-v2 on the same GPU (frames/s):

| scene | context | Newton RTX (viewer) | Newton RTX (tiled) RTPT | Minimal | PBR real-time | Newton (Warp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 4096 @ 64² | ~82-94 | 74k | 449k | 1.85M | 1.95M |
| material_grid | 1024 @ 128² | | 22k | 146k | 583k | 608k |
| damaged_helmet | 4096 @ 64² | | 68k | 547k | 310k | 168k |
| damaged_helmet | 1024 @ 128² | | 19.5k | 149k | 101k | 55k |
| franka_showroom | 4096 @ 64² | | 70k | 215k | 195k | 285k |
| franka_showroom | 1024 @ 128² | | 20.5k | 148k | 62.5k | 87k |

The tiled RTPT figure is close to IsaacLab's published OVRTX training throughput (57-92k at
8192 envs, 64²). Compared with the single-camera viewer, it is about 800× faster.

## Time breakdown (synchronised, ms per batch)

`step` is the blocking `Renderer.step` call. It returns before the GPU finishes, so most of
the GPU render time shows up in `readback`, which covers the stream wait, the map and the
split kernel. Cases shown at 64², RTPT:

| case | update | step | readback | render() total |
| --- | ---: | ---: | ---: | ---: |
| material_grid, N=1 | 1.0 | 2.7 | 4.2 | 6.1 |
| material_grid, N=1024 | 153 | 60 | 19 | 19.6 |
| material_grid, N=4096 | 727 | 282 | 49 | 55 |
| damaged_helmet, N=4096 | 24 | 21 | 57 | 60 |
| franka_showroom, N=4096 | 1735 | 1373 | 53 | 58 |

**Scene updates are expensive.** The ovstage write floor and the stage resync on the next step
scale with the prim count of the whole stage, not with the prims that changed. The first step
after an update costs 0.3-1.4 s at 4096 worlds of a multi-shape scene, against 21 ms for the
single-mesh helmet. So `render()` is fast only for a static scene. Animated batches would need
IsaacLab's path: bound `omni:xform` attributes of every world, written from the GPU with
`read_gpu_transforms`, without per-frame `advance_write_floor().wait()`.

Setup (stage export plus RTX load):
- 3-18 s up to N = 1024;
- 16 s (helmet) to 340 s (franka_showroom) at N = 4096;
- about 7 min of RTX shader compilation on the first run in a fresh container (cached after).

GPU memory, RTPT:
- 4096 worlds @ 64²: 12-14 GB;
- 4096 worlds @ 128²: 27-28 GB (8192² atlas);
- Minimal: 5-18 GB.

No black tiles appeared at any size; the largest atlas was 8192². The >4 GiB atlas problem
reported by IsaacLab was not reached.

## Image correctness

- Every RTPT tile matches the single-camera `ViewerRTX` frame of the same scene (same lights,
  materials and shadows), including world 4095. The framing differs slightly because the
  tiled cameras use the exact MuJoCo camera matrix (with roll), while the viewer reference
  maps it to pitch/yaw.
- Tiles differ from tile 0 only by 1-spp noise: 3-11 mean absolute difference on a 0-255
  scale.

## Limits

- **Fidelity is ViewerRTX's, not the scene's.**
  - Lighting is Newton's "studio" rig (dome, distant and sphere lights), not the MJCF lights
    or HDRI.
  - Materials are UsdPreviewSurface built from Newton shape colours. DamagedHelmet renders
    untextured grey.
  - franka_showroom shows only what Newton's MJCF import keeps (robot and floor).
- **Minimal mode renders our scenes badly.** With these lights it gives black objects on an
  overexposed floor (modes 1/2/3 alike), so its numbers are a speed bound for simple shading,
  not a comparable image. IsaacLab uses it with its own lights.
- **Static scene only** (see the time breakdown). The world copies are not driven by the
  Newton state; only world 0 is.
- **Identical cameras in every world**, as in the suite. The copies would need their own
  camera poses (one bound `omni:xform` write) for jittered views.
- The ovrtx API needs RTX GPUs (RT cores) and one ViewerRTX per process. Steps are not
  CUDA-graph capturable, so bench.py times them eagerly.
- DLSS frame-generation errors in the log ("NGX CreateFeature failed") come from the
  viewer's `omni:rtx:dlss:frameGeneration` setting on this headless GPU and are harmless.
