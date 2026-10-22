# Batch / many-world rendering for embodied AI and RL

Survey date: 2026-10-22. "Verified" = read from the primary source (paper text, official docs,
repo README/PR). "Unverified" = secondary source or figure-only value read by eye.

---

## 1. High-Throughput Batch Rendering for Embodied AI (Madrona renderer)

- Authors/year: Luc Guy Rosenzweig, Brennan Shacklett, Warren Xia, Kayvon Fatahalian (Stanford), SIGGRAPH Asia 2024 Conference Papers, DOI 10.1145/3680528.3687629
- URLs: https://madrona-engine.github.io/renderer.html (project), paper PDF https://madrona-engine.github.io/madrona-renderer.pdf, code https://github.com/shacklettbp/madrona (`src/mw/device` = ray tracer, `src/render` = rasterizer)
- Read: full paper text (verified).

Key takeaways
- Two backends behind one GPU-side interface (`blink`): `brast` (Vulkan batch rasterizer, evolved from bps3d) and `btrace` (software CUDA ray tracer). No OptiX/Vulkan RT: "Our initial prototypes found these platforms demonstrate poor throughput under the load of constructing many small top-level BVHs", and datacenter GPUs (H100) have no RT cores.
- `btrace` design: persistent threads (Aila & Laine 2009), two-level BVH; BLAS per object built offline and shared by all worlds; compressed wide BVH (Ylitie 2017) nodes; TLAS per environment rebuilt from scratch every frame with LBVH (Karras 2012). All per-environment TLAS builds go in one kernel dispatch over the concatenated instance table, with env offsets as bounds.
- The instance table is a struct-of-arrays sorted by (env id, Morton code) with one Onesweep radix sort. That sort replaces the separate LBVH Morton sort. Deleted rows use env id = -1 and sort to the end, so the table is compacted by truncation.
- Benchmark: 1024 environments, resolutions 64², 128², 256², Phong lighting, RTX 4090 and H100. Scenes: Hide&Seek (primitives), MJX Barkour (depth only, about 230K tris, 30 instances), ProcTHOR (about 244K tris, about 70 instances), HSSD (about 7.6M tris, about 330 instances).
- Headline numbers (verified text): HSSD 1024 views at 128² reaches 25K frames/s aggregate on H100. The ray tracer exceeds 100K FPS per GPU on simple scenes and 10K FPS on HSSD at 128². On H100, btrace beats brast in nearly every configuration and is still 22x faster on HSSD at 256². On RTX 4090, brast wins at higher resolution and simpler scenes, while btrace is competitive on MJX Barkour and HSSD (the 4090 rasterizer is limited by the Primitive Distributor on dense geometry).
- Fig. 4 axis maxima (approximate magnitudes, read from figure text, unverified): Hide&Seek up to about 2M FPS on 4090 and about 800K on H100. ProcTHOR and MJX Barkour about 100-300K. HSSD about 30-60K.
- Time breakdown (Fig. 6): trace dominates. Sort plus LBVH build is under 10% even on HSSD. The authors suggest a higher-quality (top-down/SAH) TLAS build, parallelized across environments, plus node/index compression and shared-memory traversal stacks.
- Multi-view (1-8 views per env): btrace amortizes the TLAS build across views and beats brast on ProcTHOR beyond 2 views.
- blink vs bps3d (64²): Hide&Seek 1056K vs 589K FPS (1.8x). MJX Barkour 111K vs 76K FPS (1.5x).
- Authors say btrace has "only basic GPU ray tracing optimizations", so there is headroom.

## 2. llGuy/madrona_benchmark (benchmark repo for the paper)

- URL: https://github.com/llGuy/madrona_benchmark (README read, verified as stated there)
- README table (Habitat/HSSD about 7M tris per scene):

  | GPU | Res | Ray tracer | Rasterizer |
  |---|---|---|---|
  | RTX 4090 | 64² | 60K | 15K |
  | RTX 4090 | 128² | 30K | 15K |
  | RTX 4090 | 256² | 13K | 15K |
  | H100 | 64² | 60K | 400 |
  | H100 | 128² | 30K | 400 |
  | H100 | 256² | 13K | 400 |

- The RT throughput is identical on 4090 and H100 (unverified whether that is a copy-paste artifact). The README omits world count. The H100 rasterizer (400 FPS) is catastrophically slow, which confirms that on Hopper the hardware raster path is not an option.
- From 64² to 256² (16x the pixels), FPS falls only 4.6x. Per-frame fixed cost (TLAS build, per-view setup) dominates at low resolution.

## 3. Large Batch Simulation for Deep RL (BPS / bps3d)

- Authors/year: Brennan Shacklett, Erik Wijmans, Aleksei Petrenko, Manolis Savva, Dhruv Batra, Vladlen Koltun, Kayvon Fatahalian, ICLR 2021. arXiv 2103.07013
- URLs: https://arxiv.org/abs/2103.07013, https://graphics.stanford.edu/projects/bps3D
- One render request for the whole batch of N envs, each observation a tile in a large frame. K unique scenes resident and shared across N envs (N/K ≤ 32), with assets rotated asynchronously. GPU-compute frustum culling pipelined with raster.
- 64² (SE-ResNet9) or 128² (ResNet50, rendered at 256² and downsampled). N=1024 per GPU.
- End-to-end training FPS, single GPU (Table 1): depth 19,900 (RTX 3090), 12,900 (2080Ti), 12,600 (V100). RGB 13,300 / 8,400 / 9,000. That is about 110x (depth) and 95x (RGB) over the Habitat baseline (wijmans20).
- 8 GPUs: 72K FPS depth (8x 2080Ti), 37.8K FPS RGB (8x V100). Renderer alone is about 23K FPS RGB on a 3090 at the higher resolution (secondary summary, unverified detail).

## 4. MuJoCo Playground (Madrona batch renderer inside MJX)

- Authors/year: Zakka, Tabanpour, Liao, Haiderbhai, et al. (Google DeepMind / Berkeley / others), 2025. arXiv 2502.08844
- URLs: https://arxiv.org/html/2502.08844v1, https://playground.mujoco.org
- Uses Madrona's software CUDA ray tracer through JAX custom calls (jit/vmap compatible). The Vulkan rasterizer is not used.
- Table X (RTX 4090, verified):
  - CartpoleBalance: physics only 1.37M steps/s, with pixels 403K, with inference 341K, with PPO training 31.3K.
  - PandaPickCubeCartesian: physics only 64K, with pixels 36.9K, with inference 36.0K, with PPO training 15.6K.
- Table XI breakdown per env-step:
  - Cartpole: physics 2%, render 6% (1.75 µs/frame), inference 1%, policy update 91%.
  - Panda: physics 24%, render 18% (11.5 µs), inference 1%, update 57%.
  - The bottleneck has moved to the learner (CNN update).
- Pixel pick-cube sim-to-real: 64×64 RGB, 1024 envs, trained in about 10 min on one RTX 4090, 12/12 real successes.
- Supports textures, shadows, lighting, and material domain randomization. Moving lights, deformables, and heightfields were unsupported at that time.

## 5. MuJoCo Warp (MJWarp) batch renderer

- Source: google-deepmind/mujoco_warp README, PR #1113 "mjwarp-render" by StafaH (merged 2026-02-06), docs https://mujoco.readthedocs.io/en/latest/mjwarp/, discussion #1337 (May 2026)
- API: `create_render_context(mjm, nworld, cam_res, render_rgb/depth, use_textures, use_shadows, enabled_geom_groups, cam_active, ...)`, then `refit_bvh(m,d,rc)` and `render(m,d,rc)`. Outputs have shape `(nworld, npixel)`, with RGB packed in uint32 and per-camera `rgb_adr`/`depth_adr`.
- "Single-hit raycaster aimed at throughput and low fidelity". Lambertian diffuse, point and directional lights, shadows, CUDA-texture bilinear sampling (needs Warp ≥1.12), heightfields, flex, 3DGS (≤32 sorted hits per ray, transmittance cutoff 0.005). Per-world domain randomization of material and light fields.
- Explicitly not supported: GI, physically based materials, SH>0 for splats. That is our gap to fill.
- Performance claims: "millions of frames per second" (no setup given, unverified). Cost "scales linearly with resolution and number of cameras". Meshes cost much more than primitives ("users should use primitives over visual meshes"). High-resolution performance work is "WIP".
- Maintainer (StafaH, #1337): the goal was "high throughput but lower fidelity". Custom render kernels with the same signature can fill the output buffers. Newton did this for Gaussian splats. Contributions are welcome if discussed first. This is our integration point.
- Benchmark tool: `mjwarp-testspeed <scene> --function=render`.

## 6. Newton `SensorTiledCamera` / `SensorCamera` (Warp ray-traced)

- URL: https://newton-physics.github.io/newton/latest/api/_generated/newton.sensors.SensorTiledCamera.html (deprecated in 1.7 in favor of `SensorCamera`)
- Multi-world raytraced Warp renderer. Shape and particle BVHs are built at `finalize()` and refit via `model.bvh_refit_shapes(state)`.
- Outputs: color (packed uint32), hdr_color, depth, forward_depth, normal, albedo, shape_index, each `(world, camera, H, W)`.
- sRGB to linear texture handling, ambient plus directional/spot lights, per-light shadows, cubic/triplanar projection for UV-less shapes.
- Perf knobs: `render_order` = PIXEL_PRIORITY (default) / VIEW_PRIORITY / TILED (tile_width/height), `enable_fast_math`, `kernel_block_dim`. No published numbers. The thread-to-pixel mapping options are a useful reference for coherence experiments.
- Isaac Lab 3.x exposes it as `NewtonWarpRendererCfg`. Isaac docs call it "Warp rasterization", but it is in fact ray traced per the Newton docs. Supported outputs: RGB, depth, normals, albedo, segmentation, but no motion vectors.

## 7. Isaac Lab tiled rendering

- Sources: Isaac Lab camera docs (main), Isaac Lab paper arXiv 2511.04831 (2025)
- TiledCamera packs all env cameras as tiles of one RTX framebuffer and returns `(N,H,W,C)` GPU tensors. Outputs include rgb, rgb_hdr, depth, normals, albedo, motion vectors, and semantic/instance segmentation.
- Paper (RTX Pro 6000): the per-env USD camera hits OOM above 48 cameras. Tiled and RayCaster cameras scale to thousands of envs. RayCaster (Warp mesh raycast) is faster at low resolution, while tiled RTX is faster at high resolution and scales better multi-GPU. DLSS upscaling is used for RGB. No exact FPS in text.
- About 512 cameras is recommended on 4090-class GPUs for RTX (secondary, unverified).

## 8. ManiSkill3 (SAPIEN parallel rendering)

- Tao et al., 2024/2025 (RSS 2025). arXiv 2410.00425. Benchmarks at https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html
- Rasterization renderer (SAPIEN, Vulkan) parallelized across envs. Its ray-tracing mode is not parallelized.
- Up to "30,000+ FPS" sim+render (RGBD+seg, RTX 4090). 2-3x lower GPU memory than other simulators. At 128 envs it uses 3.5 GB vs Isaac Lab's 14.1 GB, and Isaac Lab OOMs beyond 128 in that setup.
- ReplicaCAD room-scale scenes: about 2000+ FPS with rendering. Benchmark CSVs are in the ManiSkill repo (not read).

## 9. Genesis BatchRenderer

- Docs: https://genesis-world.readthedocs.io (BatchRenderer page)
- Uses the `gs-madrona` package (fork of Madrona). `use_rasterizer=True/False`; the docs call the non-raster mode "path tracing" (unverified: it is likely Madrona's btrace ray tracer). Lights are added at runtime. No numbers published.

## 10. PyBatchRender (context)

- arXiv 2601.01288 (Jan 2026). Panda3D-based. Claims >1M FPS on simple scenes, "rivaling Madrona". Abstract only, no setup details (unverified).

---

## Implications for our renderer (ranked by expected impact)

1. **The software BVH ray tracer is the right architecture on GH200.** Madrona measured H100 raster at about 400 FPS vs 13-60K FPS for software RT on HSSD. Keep the megakernel software RT and do not pursue a raster path on Hopper.
2. **Two-level BVH with shared BLAS and per-world TLAS.** Build BLAS once per mesh asset, shared by all worlds. For MuJoCo, rigid geoms map naturally to instances, so the TLAS is tiny (tens to hundreds of geoms). Rebuild or refit it every step with one batched kernel over all worlds. Madrona found that sort plus build is under 10% of frame time, which suggests a higher-quality per-world TLAS build (binned SAH, one block per world) is affordable.
3. **Fixed per-frame overhead dominates at 64-128².** FPS drops only 4.6x for 16x the pixels in Madrona. Minimize launches (megakernel, CUDA graph capture), and fuse TLAS build, camera setup, and trace where possible. Expect our sweet spot to look like Madrona's: about 1.75 µs/frame for Cartpole-like scenes at 64² on a 4090.
4. **Analytic primitives in the TLAS leaves.** MJWarp notes that meshes are much costlier than primitives. Intersect MuJoCo spheres, capsules, boxes, planes, and cylinders analytically and keep triangles only for mesh geoms.
5. **Amortize across views.** Multiple cameras per world should share the world TLAS build. Assign threads per (world, camera, pixel tile).
6. **Interface.** Match the MJWarp render-kernel signature (`(nworld, npixel)` packed buffers, `rgb_adr`, `cam_active`, geom groups) so we drop in as a custom render function (maintainer-endorsed path). Also offer HDR float, depth, normal, albedo, and segmentation like Newton/Isaac.
7. **Measure the way the field does.** Report aggregate FPS at 64/128/256² with 1024 worlds, plus the Playground Cartpole and Panda pixel envs, so numbers compare directly with Madrona, Playground Table X, and ManiSkill3.
8. **Expect the RL loop to be learner-bound** (91% policy update for pixel Cartpole). A PBR tier costing 2-5x the Lambert renderer may still be under 20% of the training step. Make that argument with measured breakdowns.
9. **Test thread-to-pixel mapping options** (Newton offers pixel/view/tiled orders). Tiled (8x4 or 8x8 pixel tiles per warp) should improve ray coherence and texture-cache locality.
