# CPU rendering

`benchmarks/cpu_bench.py` (raw rows: `results/cpu.jsonl`)

```bash
python benchmarks/cpu_bench.py --scene damaged_helmet --res 64 --nworld 32 --workers 1 8 72 [--mode process]
python benchmarks/cpu_bench.py --scene damaged_helmet --res 64 --renderer pbr_path --nworld 2 --workers 72
python benchmarks/cpu_bench.py --parity --scene damaged_helmet --res 64     # CPU vs GPU images
```

The renderer runs unmodified on Warp's CPU device (`wp.ScopedDevice("cpu")`). The setup matches
`benchmarks/suite.py`: one camera per world, identical cameras, the HDR environment
`studio_small_09_1k.hdr` for our tiers (franka_showroom uses its authored lights) and shadows on
for MuJoCo Warp. There are no CUDA graphs on the CPU, so launches are timed eagerly. Each worker
renders back to back for 3–4 s after a start barrier.

Machines (x86 is the main, complete table; the GH200 node run is partial):
- **GH200 node**: 4× 72-core Grace (Neoverse V2, aarch64, no SMT, one NUMA domain per CPU) and
  4× GH200. The GPU numbers below use one of the four GH200s.
- **x86**: a Hugging Face `cpu-performance` job, AMD EPYC 7R13 (16 cores / 32 threads, 2 NUMA
  domains, KVM).

Warp 1.18.0, MuJoCo 3.17, Mitsuba 3.9.1.

## How Warp runs CPU kernels: one launch, one core

In Warp 1.18 a CPU launch calls the generated `<kernel>_cpu_forward`. That function is a serial
`for (task_index = 0; task_index < dim.size; ++task_index)` loop on the calling thread
(`warp/_src/codegen.py`, `cpu_module_template_forward`; `wp_cpu_launch_kernel` in
`native/warp.cpp`). There is no thread pool. `wp.config.enable_cpu_blocks` runs `block_dim > 1`
as cooperative fibers, but still on one thread. Throughput is therefore flat in `nworld`: on one
core, DamagedHelmet + envmap runs at 149 frames/s with both 16 and 64 worlds.

Using every core takes one renderer per core, each owning a slice of the worlds, the way one
simulation worker per core would:

- `--mode thread` (default): one process with W Python threads, each with its own `Data` and
  render context; the `Model` is shared. Warp's launch goes through ctypes, which releases the
  GIL, so the kernels of different threads run concurrently. Threads are pinned with
  `sched_setaffinity` on their native thread id.
- `--mode process`: W processes pinned to their cores, like `benchmarks/multigpu.py`. The first
  worker compiles alone, and the others then load Warp's kernel cache. Workers start with
  `CUDA_VISIBLE_DEVICES=""`: on a GPU machine, hundreds of processes each creating a CUDA context
  serialise in the driver for minutes.

Placement is packed: NUMA domain by domain, and physical cores before SMT siblings.

## Scaling (real-time tier, 64², 32 worlds per worker, GH200 node)

| cores | helmet frames/s | per core | efficiency | material_grid frames/s | per core | efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 149 | 149 | 1.00 | 422 | 422 | 1.00 |
| 2 | 299 | 149 | 1.00 | 844 | 422 | 1.00 |
| 4 | 596 | 149 | 1.00 | 1.7k | 421 | 1.00 |
| 8 | 1.2k | 149 | 1.00 | 3.4k | 421 | 1.00 |
| 16 | 2.4k | 149 | 1.00 | 6.7k | 419 | 0.99 |
| 32 | 4.5k | 139 | 0.93 | 13k | 394 | 0.93 |
| 48 | 6.2k | 129 | 0.87 | 18k | 366 | 0.87 |
| 64 | 7.7k | 120 | 0.81 | 22k | 340 | 0.81 |
| 72 (1 Grace) | 8.4k | 116 | 0.78 | 24k | 332 | 0.79 |
| 144 (2 Grace) | 17k | 119 | 0.80 | 48k | 336 | 0.80 |
| 216 (3 Grace) | 26k | 118 | 0.79 | 73k | 338 | 0.80 |
| 288 (4 Grace) | 34k | 117 | 0.79 | 81k (threads) / **98k (processes)** | 280 / 341 | 0.66 / 0.81 |

- Scaling is linear up to 16 cores. Efficiency then falls to about 0.78 at a full Grace; the
  shared memory system and L3 slice are the likely cause. Adding Grace CPUs does not lose
  anything more: each NUMA domain adds the same per-core rate.
- Threads and processes are equal up to 72 cores (helmet 8.37k vs 8.28k; grid 23.9k vs 23.7k).
- At 288 threads, the Python side of each launch (holding the GIL) starts to bound cheap steps.
  material_grid gets 81k frames/s with threads vs 98k with processes. The path tier and MuJoCo
  Warp at 288 cores are hit too (more launches per frame; see the 288-core table). Use threads
  within one CPU socket, and processes, or more worlds per worker, beyond it.
- x86, 16 cores / 32 threads:

  | workers | helmet, threads | helmet, processes | grid, threads | grid, processes |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 126 | 126 | 347 | 349 |
  | 2 | 252 | 252 | 711 | 686 |
  | 4 | 499 | 501 | 1.4k | 1.4k |
  | 8 | 1.1k | 999 | 2.8k | 2.8k |
  | 16 | 2.0k | 2.0k | 5.6k | 5.5k |
  | 24 | 2.3k | 2.2k | 6.1k | 5.8k |
  | 32 | 2.5k | 2.4k | 6.6k | 6.1k |

## Frames/s per scene: x86, 16 cores / 32 threads (main table)

All rows were measured on the same `cpu-performance` machine; each cell is the best of thread and
process workers. Our tiers and MuJoCo Warp use 32 worlds per worker at 64² and 8 at 128² (path
tier: 2 / 1). "32" means all 32 hardware threads (16 physical cores with SMT).

| scene | res | real-time, 1 | real-time, 32 | path, 1 | path, 32 | MuJoCo Warp, 1 | MuJoCo Warp, 32 | MuJoCo GL, 1 | MuJoCo GL, 32 | Mitsuba, 1 | Mitsuba, 32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 64² | 349 | 6.6k | 13.3 | 278 | 1.2k | 21k | 19.7 | 216 | 13.0 | 15.3 |
| material_grid | 128² | 88 | 1.7k | 3.4 | 69 | 294 | 5.5k | 20.2 | 261 | 7.6 | 14.3 |
| primitives | 64² | 165 | 3.1k | 7.7 | 153 | 641 | 11k | 10.5 | 136 | 21.1 | 40.7 |
| primitives | 128² | 41 | 771 | 2.0 | 39 | 158 | 2.8k | 8.9 | 127 | 7.3 | 31.1 |
| damaged_helmet | 64² | 126 | 2.5k | 4.1 | 105 | 240 | 5.0k | 19.4 | 224 | 15.0 | 83.7 |
| damaged_helmet | 128² | 33 | 652 | 1.1 | 26 | 64 | 1.3k | 22.2 | 309 | 4.4 | 40.8 |
| anymal_c | 64² | 139 | 2.7k | 4.9 | 112 | 229 | 5.1k | 4.1 | 61 | 7.9 | 10.5 |
| anymal_c | 128² | 36 | 691 | 1.3 | 28 | 60 | 1.2k | 4.0 | 58 | 4.3 | 9.7 |
| franka_showroom | 64² | 173 | 3.3k | 8.8 | 195 | 196 | 4.2k | 2.3 | 35 | 9.8 | 10.9 |
| franka_showroom | 128² | 44 | 826 | 2.2 | 51 | 53 | 1.0k | 2.3 | 36 | 7.1 | 10.8 |

Baselines:
- **MuJoCo's OpenGL renderer** runs on Mesa llvmpipe (software rasteriser) through EGL; OSMesa
  was not used. It renders one image at a time, with one process per core and
  `LP_NUM_THREADS=1`.
- **Mitsuba 3** runs `llvm_ad_rgb` (multithreaded, vectorised LLVM JIT) with the path integrator
  at 16 spp, `max_depth=5`, our BRDF and environment (`benchmarks/references`). It renders one
  image at a time on 1 or all threads. It needs a recent libLLVM: with Debian 12's LLVM 15 its
  JIT fails ("Cannot select: fminimum"); Debian 13's LLVM 19 works.

Findings:
- **Real-time tier vs MuJoCo's OpenGL renderer on the CPU.** Ours is 11–94× faster on all
  threads: 11× on the helmet, 30× on material_grid, 44× on anymal_c, 94× on franka_showroom. The
  software rasteriser pays a fixed per-frame cost (scene update, shadow maps, readback), so it
  barely depends on resolution and grows with the number of lights and shadow casters. It gets
  2.3 frames/s per process on franka_showroom, which has several shadow-casting lights. Our tier
  adds PBR shading and image-based lighting.
- **Path tier vs Mitsuba, both at 16 spp.** On one core, Mitsuba is 1.0–3.7× faster: 3.7× on the
  helmet, 2.7× on primitives, 1.6× on anymal_c, about 1× on material_grid and franka_showroom.
  Mitsuba vectorises 8-wide with AVX2, while a Warp CPU kernel runs its threads one after
  another. Rendering one small image at a time, though, Mitsuba stops scaling after a few threads.
  On all 32 threads our tier is ahead: 1.25× on the helmet, 3.8× on primitives, 11× on anymal_c,
  18× on material_grid and franka_showroom.
- **Threads vs processes** at 32 workers: threads are 5–25% faster here (no GPU, one shared Model
  and environment). On the GH200 node they are equal up to 72 cores.
- **SMT.** Scaling is linear to 16 workers (one per physical core). The 16 extra hardware
  threads add 1.1–1.3×.
- **Per core**, the EPYC (Zen 3, KVM) gets 126 frames/s on the helmet vs 149 for a Grace core,
  and 349 vs 422 on material_grid.

## Frames/s per scene: GH200 node (second machine, partial)

Real-time tier with 32 worlds per worker at 64² and 8 at 128²; path tier (16 spp, denoised) with
2 / 1. Thread workers unless marked. GPU: one GH200 with `benchmarks/bench.py` at suite batch
sizes (4096 worlds at 64², 1024 at 128²), using CUDA graphs.

| scene | res | real-time, 1 core | real-time, 72 | real-time, 288 | path, 1 core | path, 72 | path, 288 | MuJoCo Warp, 1 | MuJoCo Warp, 72 | MuJoCo Warp, 288 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 64² | 422 | 24k | 98k (processes) | 14.8 | 848 | 2.3k | 1.4k | 79k | 137k |
| material_grid | 128² | 106 | 6.0k | 20k | 3.75 | 215 | 845 | 342 | 18k | 34k |
| primitives | 64² | 207 | 12k | 48k | 8.6 | 485 | 1.7k | 732 | 42k | 120k |
| primitives | 128² | 52 | 2.9k | 12k | 2.2 | 123 | 493 | 181 | 10k | 28k |
| damaged_helmet | 64² | 149 | 8.3k | 34k | 4.9 | 263 | 585 | 285 | 16k | 65k |
| damaged_helmet | 128² | 39 | 2.2k | 8.9k | 1.25 | 69 | 271 | 75 | 4.2k | 17k |
| anymal_c | 64² | 166 | 9.1k | 37k | – | – | – | 274 | 15k | 61k |

### CPU vs GPU

| scene | res | GPU real-time | GPU path | GPU MuJoCo Warp | GPU / 72-core Grace (real-time) | GPU / whole node (real-time) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 64² | 883k | 5.0k | 1.49M | 37× | 9.0× |
| material_grid | 128² | 281k | 1.4k | 417k | 47× | 14× |
| primitives | 64² | 414k | 2.3k | 653k | 35× | 8.6× |
| primitives | 128² | 121k | 668 | 186k | 42× | 10× |
| damaged_helmet | 64² | 176k | 1.7k | 109k | 21× | 5.1× |
| damaged_helmet | 128² | 53k | 460 | 29k | 25× | 6.0× |
| anymal_c | 64² | 57k | 1.1k | 59k | 6.3× | 1.5× |
| anymal_c | 128² | 19k | 294 | 17k | – | – |
| franka_showroom | 64² | 55k | 2.4k | 53k | – | – |
| franka_showroom | 128² | 18k | 737 | 15k | – | – |

- One 72-core Grace delivers 3–16% of one GH200 on the real-time tier, and the whole node
  (288 cores) 7–65%. The gap is smallest on anymal_c, the scene the GPU finds hardest at 64²
  (57k frames/s vs 176k for the helmet), while the CPU renders it about as fast as the helmet.
- The path tier narrows the gap. The GPU is 6.4× faster than one Grace on the helmet (1.7k vs
  263 frames/s) and 5.9× on material_grid, against 21× and 37× for the real-time tier. The GPU
  path tier runs at 0.6–2% of its real-time rate at 64², the CPU path tier at about 3%.

### Missing GH200-node rows

The node run was stopped before it finished. Not measured on Grace: anymal_c at 128², the
anymal_c path tier, franka_showroom, MuJoCo's OpenGL renderer and Mitsuba (all of these are in
the x86 table), and a 288-core process-mode pass. In the 288-core column every cell except
material_grid 64² uses threads and may be GIL-bound, so treat those cells as lower bounds (most
clearly the path tier and MuJoCo Warp on light scenes).

## Parity (CPU vs GPU, world 0, GH200 node)

| scene | res | real-time max / >1/255 / PSNR | path max / >1/255 / PSNR | MuJoCo Warp max / >1/255 / PSNR |
| --- | --- | --- | --- | --- |
| material_grid | 64² | 1/255, 0.00%, 69.5 dB | 5/255, 0.42%, 63.0 dB | 1/255, 0.00%, 80.0 dB |
| material_grid | 128² | 1/255, 0.00%, 69.3 dB | 56/255, 0.31%, 55.3 dB | 1/255, 0.00%, 83.6 dB |
| damaged_helmet | 64² | 1/255, 0.00%, 64.3 dB | 7/255, 0.42%, 61.3 dB | 1/255, 0.00%, 72.5 dB |
| damaged_helmet | 128² | 2/255, 0.05%, 64.1 dB | 33/255, 0.13%, 56.6 dB | 1/255, 0.00%, 73.3 dB |
| primitives | 64² | 1/255, 0.00%, 65.8 dB | 14/255, 0.32%, 60.0 dB | 1/255, 0.00%, 80.0 dB |
| primitives | 128² | 1/255, 0.00%, 66.4 dB | 45/255, 0.55%, 56.2 dB | 1/255, 0.00%, 77.5 dB |

The real-time tier matches the GPU to 8-bit rounding. The path tier uses the same random
streams. In 0.1–0.6% of pixels, floating-point differences send a path down a different branch.
The result is isolated outlier pixels, with mean absolute differences of 0.02/255.

## Startup: environment prefilter, not compilation

Times are for the first worker; "build" is creating the render context, "first call" is the
first `render`.

| | build | first call |
| --- | ---: | ---: |
| real-time tier + envmap, cold kernel cache (72-core Grace, shared) | 35–37 s | 1.2–1.7 s |
| path tier + envmap, cold cache | 35 s | 2.2–3.1 s |
| MuJoCo Warp, cold cache | 2.4–2.8 s | 0.7–0.9 s |
| real-time tier + envmap, warm cache (GH200 node) | 32.5–34 s | 0.1–0.3 s |
| x86, cold cache | 35–38 s | 2.5–5 s |

Warp compiles the CPU kernels in seconds. Almost all of the ~35 s goes to prefiltering the 1k
HDR environment (`envmap.EnvironmentMap`): Warp kernels that run serially on one core, once per
render context. `cpu_bench.py` shares the prefiltered maps between the render contexts of a
process, so 288 thread workers pay the cost once. Process workers pay it once each, in parallel.

When compiling `ibl.py` for the CPU, Clang warns `implicit conversion from 'long' to 'const
wp::int32' changes value from 4042322160 to ...`. These are the bit-reversal masks, already
written as `wp.uint32(0xF0F0F0F0)`. Warp's codegen declares the literal as an int32 before the
cast, so the bit pattern, and therefore the result, is unchanged. The warning is harmless and
was left alone.

## Recommendations

- **CI and tests on CPU-only runners: viable.** Images match the GPU to 1/255 for the
  real-time tier. Expect about 35 s of startup per render context with an HDR environment;
  tests that skip `envmap` start in a few seconds.
- **Small batches and laptops: usable.** One core renders 100–400 real-time frames/s at 64²
  (25–100 at 128²). An 8-core laptop gets about 1–3k frames/s with one thread per core. That is
  enough for debugging, visualisation and small RL runs. Path-traced frames take 0.07–0.25 s each
  per core at 64².
- **Large-scale RL: use a GPU.** A whole 4-CPU node is 1.5–14× slower than one of its GPUs on
  the real-time tier.
- **Scaling recipe:** one worker per physical core, each with its own render context and
  `nworld` large enough for steps of at least about 50 ms. Use threads within one CPU socket and
  processes beyond it; with processes on a GPU machine, hide the GPUs.
- **Startup:** precompute or cache the prefiltered environment, or parallelise the prefilter
  kernels, to remove the ~35 s per context.
