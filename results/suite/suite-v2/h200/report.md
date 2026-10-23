# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

## NVIDIA H200

NVIDIA H200, 580.178.04, 143771 MiB, 9.0. Linux-6.12.105+-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `e7ce6c3a774d`.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 30.8k | 314 | 17.6k | **38.3k** | n/a | 36.6k | 65 | 24 |
| primitives | 18.6k | 158 | 11.2k | 20.7k | n/a | **29.6k** | 28 | 68 |
| damaged_helmet | 4.6k | 84 | 4.4k | 4.5k | n/a | **4.8k** | 86 | 91 |
| a_beautiful_game | 2.2k | 42 |  |  | n/a | **3.0k** | 4 | 19 |
| anymal_c | 2.3k | 47 | 1.9k | 2.8k | n/a | **3.3k** | 19 | 18 |
| franka_showroom | 1.3k | 41 | 1.5k | **1.9k** | n/a | 1.8k | 8 | 16 |
| isaac_franka_cabinet | 1.7k | 29 | 1.5k | 1.2k | n/a | **2.3k** | 22 | 20 |
| isaac_seattle_lab_table | 558 | 7 | 517 | **4.6k** | n/a | 565 | 35 | 189 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 908.9k | 5.1k | 704.0k | 651.3k | **1.51M** |
| primitives | 400.1k | 2.3k | 485.3k | 581.8k | **646.9k** |
| damaged_helmet | 172.2k | 1.6k | **178.9k** | 114.8k | 108.4k |
| a_beautiful_game | 125.3k | 1.2k | **128.4k** | 56.7k | 96.6k |
| anymal_c | 56.2k | 1.1k | 48.6k | **79.8k** | 58.0k |
| franka_showroom | 51.9k | 2.1k | 44.3k | **118.9k** | 51.7k |
| isaac_franka_cabinet | 136.9k | 1.3k | **143.0k** | 76.4k | 76.6k |
| isaac_seattle_lab_table | 48.5k | 415 | 49.7k | **180.4k** | 12.6k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 285.4k | 1.4k | 220.8k | 251.5k | n/a | **420.6k** | 79 | 24 |
| primitives | 118.0k | 667 | 144.0k | **209.0k** | n/a | 183.8k | 32 | 85 |
| damaged_helmet | 52.4k | 450 | **54.1k** | 40.4k | n/a | 28.9k | 80 | 185 |
| a_beautiful_game | **38.2k** | 332 |  |  |  | 25.6k | 3 | 19 |
| anymal_c | 18.8k | 294 | 16.1k | **30.7k** | n/a | 17.1k | 17 | 21 |
| franka_showroom | 17.1k | 664 | 14.7k | **40.9k** | n/a | 15.0k | 8 | 19 |
| isaac_franka_cabinet | 43.4k | 349 | **46.1k** | 26.0k | n/a | 20.1k | 20 | 20 |
| isaac_seattle_lab_table | 16.4k | 127 | 17.4k | **62.2k** | n/a | 4.0k | 31 | 240 |

### Gallery

MuJoCo GL | MuJoCo Warp | Newton | Newton RTX | PBR real-time | PBR path-traced | Mitsuba (reference).

**a_beautiful_game**

![a_beautiful_game](gallery/a_beautiful_game.png)

**anymal_c**

![anymal_c](gallery/anymal_c.png)

**damaged_helmet**

![damaged_helmet](gallery/damaged_helmet.png)

**franka_showroom**

![franka_showroom](gallery/franka_showroom.png)

**material_grid**

![material_grid](gallery/material_grid.png)

**primitives**

![primitives](gallery/primitives.png)

