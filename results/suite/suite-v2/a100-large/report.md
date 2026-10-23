# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

## NVIDIA A100-SXM4-80GB

NVIDIA A100-SXM4-80GB, 580.178.04, 81920 MiB, 8.0. Linux-6.12.103-127.188.amzn2023.x86_64-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `e7ce6c3a774d`.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 22.9k | 225 | 13.3k | 25.0k | n/a | **28.4k** | 1.6k | 17 |
| primitives | 12.1k | 123 | 8.5k | 13.5k | n/a | **21.9k** | 1.4k | 54 |
| damaged_helmet | 3.2k | 62 | 3.1k | **3.3k** | n/a | 3.0k | 2.2k | 72 |
| a_beautiful_game | 1.5k | 32 |  |  | n/a | **1.9k** | 672 | 13 |
| anymal_c | 1.8k | 40 | 1.4k | 2.1k | n/a | **2.4k** | 900 | 13 |
| franka_showroom | 842 | 30 | 1.0k | **1.5k** | n/a | 1.5k | 658 | 12 |
| isaac_franka_cabinet | 1.2k | 21 | 1.1k | 924 | n/a | **1.7k** | 1.6k | 11 |
| isaac_seattle_lab_table | 391 | 5 | 379 | **3.5k** | n/a | 426 | 1.7k | 129 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 551.8k | 3.6k | 431.1k | 398.5k | **900.8k** |
| primitives | 258.9k | 1.7k | 295.5k | 348.9k | **415.7k** |
| damaged_helmet | **114.1k** | 1.1k | 113.9k | 68.2k | 65.0k |
| a_beautiful_game | **79.3k** | 681 | 77.4k | 35.1k | 57.3k |
| anymal_c | 43.9k | 736 | 37.8k | **47.8k** | 35.9k |
| franka_showroom | 41.6k | 1.3k | 35.2k | **72.4k** | 34.8k |
| isaac_franka_cabinet | 83.5k | 800 | **85.7k** | 46.1k | 47.0k |
| isaac_seattle_lab_table | 27.4k | 241 | 28.3k | **109.0k** | 8.1k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 170.4k | 1.0k | 132.5k | 150.5k | n/a | **257.7k** | 2.3k | 18 |
| primitives | 74.1k | 508 | 83.9k | **125.7k** | n/a | 118.6k | 1.7k | 62 |
| damaged_helmet | 35.3k | 313 | **35.3k** | 23.3k | n/a | 17.2k | 2.9k | 136 |
| a_beautiful_game | **22.1k** | 188 |  |  |  | 14.8k | 862 | 13 |
| anymal_c | 14.6k | 208 | 12.2k | **17.8k** | n/a | 10.4k | 988 | 13 |
| franka_showroom | 13.5k | 405 | 11.5k | **25.4k** | n/a | 9.5k | 739 | 12 |
| isaac_franka_cabinet | 25.4k | 224 | **26.3k** | 16.0k | n/a | 12.2k | 2.1k | 12 |
| isaac_seattle_lab_table | 9.4k | 74 | 9.7k | **37.2k** | n/a | 2.4k | 2.4k | 158 |

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

