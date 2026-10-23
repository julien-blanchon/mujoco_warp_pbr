# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

## NVIDIA RTX PRO 6000 Blackwell Server Edition

NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.178.04, 97887 MiB, 12.0. Linux-6.12.103-127.188.amzn2023.x86_64-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `f137f228d868`.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 38.1k | 3.3k | 519 | 21.8k | **67.7k** | 82 | 43.4k | 3.3k | 22 |
| primitives | 28.3k | 3.3k | 402 | 15.4k | **46.2k** | 80 | 38.4k | 2.8k | 70 |
| damaged_helmet | **5.0k** | 3.5k | 82 | 4.9k | 3.3k | 91 | 3.7k | 4.4k | 218 |
| a_beautiful_game | 2.8k | 3.5k | 60 |  |  | 83 | **3.7k** | 2.4k | 18 |
| anymal_c | 2.4k | **3.6k** | 54 | 2.0k | 2.5k | 76 | 2.8k | 2.4k | 18 |
| franka_showroom | 1.2k | **3.5k** | 31 | 1.4k | 1.8k | 72 | 1.7k | 2.3k | 16 |
| isaac_franka_cabinet | 1.8k | 3.1k | 29 | 1.7k | 1.3k | 77 | 2.4k | **3.7k** | 18 |
| isaac_seattle_lab_table | 474 | 3.6k | 5 | 415 | 3.9k | 91 | 371 | **5.4k** | 229 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 1.85M | 1.86M | 12.5k | 1.30M | 1.95M | **2.54M** |
| primitives | 1.21M | 653.0k | 10.7k | 1.09M | 1.88M | **2.08M** |
| damaged_helmet | 309.6k | **1.64M** | 2.1k | 304.9k | 168.2k | 111.6k |
| a_beautiful_game | 244.2k | **1.33M** | 2.0k | 238.4k | 134.2k | 146.8k |
| anymal_c | 176.1k | **931.6k** | 1.9k | 148.4k | 176.5k | 81.6k |
| franka_showroom | 194.6k | **729.8k** | 2.8k | 169.8k | 284.8k | 77.9k |
| isaac_franka_cabinet | 262.9k | **1.37M** | 2.1k | 259.5k | 192.4k | 102.3k |
| isaac_seattle_lab_table | 55.0k | **1.66M** | 393 | 56.7k | 288.3k | 11.3k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 583.2k | 619.6k | 3.6k | 401.9k | 607.6k | 94 | **697.4k** | 5.7k | 23 |
| primitives | 340.2k | 190.3k | 2.9k | 318.9k | **552.0k** | 93 | 547.0k | 4.3k | 69 |
| damaged_helmet | 101.0k | **475.5k** | 638 | 98.5k | 55.1k | 97 | 29.4k | 8.9k | 245 |
| a_beautiful_game | 77.6k | **392.0k** | 556 |  |  |  | 37.3k | 2.8k | 18 |
| anymal_c | 61.3k | **315.2k** | 541 | 51.1k | 57.7k | 83 | 23.2k | 3.2k | 18 |
| franka_showroom | 62.5k | **263.5k** | 853 | 54.6k | 87.0k | 82 | 21.3k | 2.8k | 16 |
| isaac_franka_cabinet | 87.6k | **387.9k** | 584 | 86.9k | 56.7k | 90 | 26.5k | 4.8k | 17 |
| isaac_seattle_lab_table | 18.0k | **482.0k** | 119 | 18.3k | 95.4k | 96 | 3.6k | 7.1k | 233 |

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

