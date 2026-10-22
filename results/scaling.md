# Batch-size scaling

`python benchmarks/sweep.py --preset scaling --out results/scaling.jsonl --extra --jitter-camera`

Setup:
- one GH200 (idle during the run), real-time tier with defaults (TILED, register cap 96, auto
  block_dim), 1 spp;
- HDR environment plus one shadow-casting sun, per-world camera jitter;
- MuJoCo Warp's renderer with shadows on, same scenes;
- one camera per world.

## Model

Batch time follows a fixed latency plus a throughput term, within about 10%:

    t(nworld) = t0 + nworld / peak        frames/s = nworld / t = peak · nworld / (nworld + n½)

`n½ = t0 · peak` is the batch size reaching half the peak throughput.

| renderer | scene | res | t0 (µs) | peak (frames/s) | n½ (worlds) | n½ (Mpx) |
| --- | --- | --- | --- | --- | --- | --- |
| PBR | damaged_helmet | 64 | 175 | 177.5k | 31 | 0.13 |
| PBR | damaged_helmet | 128 | 181 | 52.4k | 9 | 0.16 |
| PBR | primitives | 64 | 33 | 318k | 11 | 0.04 |
| PBR | primitives | 128 | 32 | 106k | 3 | 0.06 |
| PBR | material_grid | 64 | 26 | 953k | 25 | 0.10 |
| PBR | material_grid | 128 | 28 | 311k | 9 | 0.14 |
| MuJoCo Warp | damaged_helmet | 64 | 130 | 110k | 14 | 0.06 |
| MuJoCo Warp | damaged_helmet | 128 | 117 | 27.1k | 3 | 0.05 |
| MuJoCo Warp | primitives | 64 | 26 | 625k | 16 | 0.07 |
| MuJoCo Warp | primitives | 128 | 28 | 175k | 5 | 0.08 |
| MuJoCo Warp | material_grid | 64 | 30 | 1.53M | 45 | 0.19 |
| MuJoCo Warp | material_grid | 128 | 29 | 434k | 12 | 0.20 |

## Measured frames/s (64²)

| nworld | helmet PBR | helmet MJW | primitives PBR | primitives MJW | grid PBR | grid MJW |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.2k | 7.2k | 21k | 32k | 34k | 31k |
| 16 | 65k | 65k | 268k | 364k | 434k | 428k |
| 256 | 159k | 104k | 290k | 586k | 866k | 1.33M |
| 4096 | 173k | 109k | 294k | 599k | 924k | 1.47M |
| 32768 | 173k | 106k | 291k | 600k | 928k | 1.47M |

## Reading

- **Saturation.** Throughput reaches 90% of peak at about 10 × n½, i.e. 1–4 M pixels per
  launch: roughly 256 worlds at 64² or 64 worlds at 128². Beyond that, time is linear in nworld
  and frames/s stays flat up to the largest batch tested (32768 worlds at 64², 134 M pixels).
  The GPU is full, so larger batches buy no extra efficiency.
- **Below saturation the launch is latency-bound.** The floor t0 is about 30 µs for primitive
  scenes and about 180 µs for the mesh scene. The mesh floor is the latency of one mesh-BVH
  traversal chain, which a small batch cannot hide.
- **Cost against resolution.** It grows sublinearly: 4× the pixels (64² to 128²) costs 3.0–3.4×
  the time, because neighbouring rays are more coherent.
- **Against MuJoCo Warp's Lambert/Phong renderer (shadows on).**
  - The PBR tier is 1.6–1.9× faster on the mesh scene at saturation.
  - It is 0.5–0.6× as fast on the primitive scene and 0.63–0.72× on the sphere grid, where
    shading (IBL, specular, textures, specular AA) outweighs traversal.
  - At very small batches (1–4 worlds) the mesh scene is latency-bound and slightly slower than
    MuJoCo Warp.
