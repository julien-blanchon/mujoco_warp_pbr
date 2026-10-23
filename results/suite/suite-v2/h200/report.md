# Benchmark suite

Camera-frames per second (one camera per world, identical cameras in every world; best per row in
bold). MuJoCo GL and Mitsuba render one image at a time, so their batch numbers are per-image
throughput; their batch_64 cell is not measured. n/a: the renderer cannot draw the scene
(Mitsuba's exporter has no capsules / cylinders) or failed on this machine.

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

