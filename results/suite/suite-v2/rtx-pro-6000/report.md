# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

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

**a_beautiful_game** ([orbit video](gallery/a_beautiful_game.mp4))

![a_beautiful_game](gallery/a_beautiful_game.png)

**anymal_c** ([orbit video](gallery/anymal_c.mp4))

![anymal_c](gallery/anymal_c.png)

**damaged_helmet** ([orbit video](gallery/damaged_helmet.mp4))

![damaged_helmet](gallery/damaged_helmet.png)

**franka_showroom** ([orbit video](gallery/franka_showroom.mp4))

![franka_showroom](gallery/franka_showroom.png)

**isaac_franka_cabinet** ([orbit video](gallery/isaac_franka_cabinet.mp4))

![isaac_franka_cabinet](gallery/isaac_franka_cabinet.png)

**isaac_seattle_lab_table** ([orbit video](gallery/isaac_seattle_lab_table.mp4))

![isaac_seattle_lab_table](gallery/isaac_seattle_lab_table.png)

**material_grid** ([orbit video](gallery/material_grid.mp4))

![material_grid](gallery/material_grid.png)

**primitives** ([orbit video](gallery/primitives.mp4))

![primitives](gallery/primitives.png)

