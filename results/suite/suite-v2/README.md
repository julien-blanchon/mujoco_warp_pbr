# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

## Across machines (PBR real-time)

| machine | Single scene: 1 world, 256² | Batch: 4096 worlds, 64² | Batch: 1024 worlds, 128² |
| --- | ---: | ---: | ---: |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 5.5k (geo. mean) | 418.1k (geo. mean) | 132.8k (geo. mean) |
| NVIDIA H200 | 4.7k (geo. mean) | 166.7k (geo. mean) | 52.2k (geo. mean) |
| NVIDIA A100-SXM4-80GB | 3.2k (geo. mean) | 114.7k (geo. mean) | 34.9k (geo. mean) |

## NVIDIA RTX PRO 6000 Blackwell Server Edition

NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.178.04, 97887 MiB, 12.0. Linux-6.12.103-127.188.amzn2023.x86_64-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `f137f228d868`.

PBR real-time, path-traced and OptiX rows, and Newton RTX (tiled), re-measured at commit `5da78fdef50c` (mesh BVHs in the authored frame, OptiX follow-up); the PBR through Newton rows predate the BVH change.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | Newton RTX (tiled) | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 38.2k | 3.2k | 523 | 21.8k | **67.7k** | 82 | 113 | 43.4k | 3.3k | 22 |
| primitives | 28.1k | 3.0k | 401 | 15.4k | **46.2k** | 80 | 113 | 38.4k | 2.8k | 70 |
| damaged_helmet | **5.0k** | 3.5k | 83 | 4.9k | 3.3k | 91 | 115 | 3.7k | 4.4k | 218 |
| a_beautiful_game | 2.8k | 3.4k | 60 |  |  | 83 |  | **3.7k** | 2.4k | 18 |
| anymal_c | 2.4k | **3.5k** | 56 | 2.0k | 2.5k | 76 | 113 | 2.8k | 2.4k | 18 |
| franka_showroom | 1.7k | **2.9k** | 53 | 1.4k | 1.8k | 72 | 113 | 1.7k | 2.3k | 16 |
| isaac_franka_cabinet | 1.9k | 3.4k | 44 | 1.7k | 1.3k | 77 | 118 | 2.4k | **3.7k** | 18 |
| isaac_seattle_lab_table | **6.4k** | 3.4k | 97 | 415 | 3.9k | 91 | 118 | 371 | 5.4k | 229 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX (tiled) | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 1.87M | 2.39M | 12.6k | 1.30M | 1.95M | 73.1k | **2.54M** |
| primitives | 1.22M | 1.05M | 10.8k | 1.09M | 1.88M | 72.8k | **2.08M** |
| damaged_helmet | 311.7k | **1.61M** | 2.2k | 304.9k | 168.2k | 70.5k | 111.6k |
| a_beautiful_game | 245.9k | **1.33M** | 2.0k | 238.4k | 134.2k | 64.9k | 146.8k |
| anymal_c | 180.3k | **926.1k** | 1.9k | 148.4k | 176.5k | 72.6k | 81.6k |
| franka_showroom | 226.1k | **795.0k** | 3.2k | 169.8k | 284.8k | 72.2k | 77.9k |
| isaac_franka_cabinet | 274.4k | **1.86M** | 2.2k | 259.5k | 192.4k | 72.3k | 102.3k |
| isaac_seattle_lab_table | 477.4k | **1.57M** | 4.1k | 56.7k | 288.3k | 71.5k | 11.3k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR real-time (OptiX) | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | Newton RTX (tiled) | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 588.1k | **745.0k** | 3.6k | 401.9k | 607.6k | 94 | 22.1k | 697.4k | 5.7k | 23 |
| primitives | 342.1k | 307.0k | 2.9k | 318.9k | **552.0k** | 93 | 20.7k | 547.0k | 4.3k | 69 |
| damaged_helmet | 101.4k | **470.8k** | 644 | 98.5k | 55.1k | 97 | 19.6k | 29.4k | 8.9k | 245 |
| a_beautiful_game | 78.1k | **387.5k** | 560 |  |  |  |  | 37.3k | 2.8k | 18 |
| anymal_c | 62.9k | **314.2k** | 554 | 51.1k | 57.7k | 83 | 20.0k | 23.2k | 3.2k | 18 |
| franka_showroom | 71.0k | **277.6k** | 983 | 54.6k | 87.0k | 82 | 20.2k | 21.3k | 2.8k | 16 |
| isaac_franka_cabinet | 91.5k | **539.5k** | 628 | 86.9k | 56.7k | 90 | 20.1k | 26.5k | 4.8k | 17 |
| isaac_seattle_lab_table | 148.4k | **475.7k** | 1.2k | 18.3k | 95.4k | 96 | 19.3k | 3.6k | 7.1k | 233 |

### Gallery

MuJoCo GL | MuJoCo Warp | Newton | Newton RTX | PBR real-time | PBR path-traced | Mitsuba (reference).

**a_beautiful_game**

![a_beautiful_game](rtx-pro-6000/gallery/a_beautiful_game.png)

**anymal_c**

![anymal_c](rtx-pro-6000/gallery/anymal_c.png)

**damaged_helmet**

![damaged_helmet](rtx-pro-6000/gallery/damaged_helmet.png)

**franka_showroom**

![franka_showroom](rtx-pro-6000/gallery/franka_showroom.png)

**material_grid**

![material_grid](rtx-pro-6000/gallery/material_grid.png)

**primitives**

![primitives](rtx-pro-6000/gallery/primitives.png)

## NVIDIA H200

NVIDIA H200, 580.178.04, 143771 MiB, 9.0. Linux-6.12.105+-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `e7ce6c3a774d`.

PBR real-time and path-traced rows re-measured at commit `5da78fdef50c` (mesh BVHs in the authored frame); the PBR through Newton rows predate it.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 30.0k | 311 | 17.6k | **38.3k** | n/a | 36.6k | 65 | 24 |
| primitives | 18.4k | 158 | 11.2k | 20.7k | n/a | **29.6k** | 28 | 68 |
| damaged_helmet | 4.7k | 84 | 4.4k | 4.5k | n/a | **4.8k** | 86 | 91 |
| a_beautiful_game | 2.2k | 42 |  |  | n/a | **3.0k** | 4 | 19 |
| anymal_c | 2.4k | 47 | 1.9k | 2.8k | n/a | **3.3k** | 19 | 18 |
| franka_showroom | 1.7k | 65 | 1.5k | **1.9k** | n/a | 1.8k | 8 | 16 |
| isaac_franka_cabinet | 1.8k | 40 | 1.5k | 1.2k | n/a | **2.3k** | 22 | 20 |
| isaac_seattle_lab_table | **5.9k** | 86 | 517 | 4.6k | n/a | 565 | 35 | 189 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 896.9k | 5.0k | 704.0k | 651.3k | **1.51M** |
| primitives | 412.7k | 2.3k | 485.3k | 581.8k | **646.9k** |
| damaged_helmet | 174.8k | 1.6k | **178.9k** | 114.8k | 108.4k |
| a_beautiful_game | 125.3k | 1.2k | **128.4k** | 56.7k | 96.6k |
| anymal_c | 56.4k | 1.1k | 48.6k | **79.8k** | 58.0k |
| franka_showroom | 55.1k | 2.4k | 44.3k | **118.9k** | 51.7k |
| isaac_franka_cabinet | 135.7k | 1.3k | **143.0k** | 76.4k | 76.6k |
| isaac_seattle_lab_table | 174.6k | 2.1k | 49.7k | **180.4k** | 12.6k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 281.6k | 1.4k | 220.8k | 251.5k | n/a | **420.6k** | 79 | 24 |
| primitives | 120.6k | 658 | 144.0k | **209.0k** | n/a | 183.8k | 32 | 85 |
| damaged_helmet | 53.3k | 452 | **54.1k** | 40.4k | n/a | 28.9k | 80 | 185 |
| a_beautiful_game | **38.1k** | 333 |  |  |  | 25.6k | 3 | 19 |
| anymal_c | 19.0k | 294 | 16.1k | **30.7k** | n/a | 17.1k | 17 | 21 |
| franka_showroom | 18.1k | 732 | 14.7k | **40.9k** | n/a | 15.0k | 8 | 19 |
| isaac_franka_cabinet | 43.2k | 358 | **46.1k** | 26.0k | n/a | 20.1k | 20 | 20 |
| isaac_seattle_lab_table | 54.0k | 582 | 17.4k | **62.2k** | n/a | 4.0k | 31 | 240 |

### Gallery

MuJoCo GL | MuJoCo Warp | Newton | Newton RTX | PBR real-time | PBR path-traced | Mitsuba (reference).

**a_beautiful_game**

![a_beautiful_game](h200/gallery/a_beautiful_game.png)

**anymal_c**

![anymal_c](h200/gallery/anymal_c.png)

**damaged_helmet**

![damaged_helmet](h200/gallery/damaged_helmet.png)

**franka_showroom**

![franka_showroom](h200/gallery/franka_showroom.png)

**material_grid**

![material_grid](h200/gallery/material_grid.png)

**primitives**

![primitives](h200/gallery/primitives.png)

## NVIDIA A100-SXM4-80GB

NVIDIA A100-SXM4-80GB, 580.178.04, 81920 MiB, 8.0. Linux-6.12.103-127.188.amzn2023.x86_64-x86_64-with-glibc2.39. warp-lang 1.18.0, mujoco 3.17.0.dev1004626286, mujoco-warp 3.16.0, newton 1.7.0, mitsuba 3.9.1, mujoco-warp-pbr 0.0.1.

Commit `e7ce6c3a774d`.

PBR real-time and path-traced rows re-measured at commit `5da78fdef50c` (mesh BVHs in the authored frame); the PBR through Newton rows predate it.

### Single scene: 1 world, 256²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 22.6k | 222 | 13.3k | 25.0k | n/a | **28.4k** | 1.6k | 17 |
| primitives | 12.0k | 121 | 8.5k | 13.5k | n/a | **21.9k** | 1.4k | 54 |
| damaged_helmet | 3.2k | 61 | 3.1k | **3.3k** | n/a | 3.0k | 2.2k | 72 |
| a_beautiful_game | 1.5k | 32 |  |  | n/a | **1.9k** | 672 | 13 |
| anymal_c | 1.7k | 40 | 1.4k | 2.1k | n/a | **2.4k** | 900 | 13 |
| franka_showroom | 1.1k | 48 | 1.0k | **1.5k** | n/a | 1.5k | 658 | 12 |
| isaac_franka_cabinet | 1.3k | 28 | 1.1k | 924 | n/a | **1.7k** | 1.6k | 11 |
| isaac_seattle_lab_table | **3.7k** | 65 | 379 | 3.5k | n/a | 426 | 1.7k | 129 |

### Batch: 4096 worlds, 64²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | MuJoCo Warp |
| --- | ---: | ---: | ---: | ---: | ---: |
| material_grid | 536.8k | 3.5k | 431.1k | 398.5k | **900.8k** |
| primitives | 258.9k | 1.7k | 295.5k | 348.9k | **415.7k** |
| damaged_helmet | **115.3k** | 1.1k | 113.9k | 68.2k | 65.0k |
| a_beautiful_game | **79.9k** | 679 | 77.4k | 35.1k | 57.3k |
| anymal_c | 45.0k | 738 | 37.8k | **47.8k** | 35.9k |
| franka_showroom | 45.1k | 1.5k | 35.2k | **72.4k** | 34.8k |
| isaac_franka_cabinet | **86.0k** | 818 | 85.7k | 46.1k | 47.0k |
| isaac_seattle_lab_table | **134.2k** | 1.6k | 28.3k | 109.0k | 8.1k |

### Batch: 1024 worlds, 128²

| scene | PBR real-time | PBR path (16 spp) | PBR via Newton | Newton | Newton RTX | MuJoCo Warp | MuJoCo GL | Mitsuba (16 spp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| material_grid | 165.9k | 1.0k | 132.5k | 150.5k | n/a | **257.7k** | 2.3k | 18 |
| primitives | 73.3k | 500 | 83.9k | **125.7k** | n/a | 118.6k | 1.7k | 62 |
| damaged_helmet | **35.5k** | 311 | 35.3k | 23.3k | n/a | 17.2k | 2.9k | 136 |
| a_beautiful_game | **22.2k** | 188 |  |  |  | 14.8k | 862 | 13 |
| anymal_c | 15.0k | 208 | 12.2k | **17.8k** | n/a | 10.4k | 988 | 13 |
| franka_showroom | 14.4k | 449 | 11.5k | **25.4k** | n/a | 9.5k | 739 | 12 |
| isaac_franka_cabinet | **26.4k** | 229 | 26.3k | 16.0k | n/a | 12.2k | 2.1k | 12 |
| isaac_seattle_lab_table | **40.5k** | 441 | 9.7k | 37.2k | n/a | 2.4k | 2.4k | 158 |

### Gallery

MuJoCo GL | MuJoCo Warp | Newton | Newton RTX | PBR real-time | PBR path-traced | Mitsuba (reference).

**a_beautiful_game**

![a_beautiful_game](a100-large/gallery/a_beautiful_game.png)

**anymal_c**

![anymal_c](a100-large/gallery/anymal_c.png)

**damaged_helmet**

![damaged_helmet](a100-large/gallery/damaged_helmet.png)

**franka_showroom**

![franka_showroom](a100-large/gallery/franka_showroom.png)

**material_grid**

![material_grid](a100-large/gallery/material_grid.png)

**primitives**

![primitives](a100-large/gallery/primitives.png)

