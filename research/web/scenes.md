# Candidate test scenes: PBR showcases, robotics, NVIDIA demo scenes, light rigs

Survey date: 2026-10-22. Sizes come from file listings, and the material and light contents of the
USD scenes were parsed from their layers. Unverified items are marked.

## NVIDIA Isaac / Newton demo scenes (USD)

**Where they live.** IsaacLab's asset root (`persistent.isaac.asset_root.default` in IsaacLab's
`apps/*.kit`) is `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2`
(`ISAAC_NUCLEUS_DIR = {root}/Isaac`). Anonymous S3 listing works.

**Licences.** Environment folders ship a CC-BY-4.0 `LICENSE`, but
`Environments/environment-supplement-LICENSE.txt` declares them "Limited Use Content: for your use
only, without modifications". Treat them as not redistributable.

Folder by folder (checked 2026-10-22 on the 6.2 listing):
- `IsaacLab/Robots/FrankaEmika/`: Apache-2.0 `LICENSE` (the USD was generated from MuJoCo
  Menagerie's panda by the MuJoCo USD converter).
- `Props/Sektion_Cabinet/`, `Props/Mounts/SeattleLabTable/`, `IsaacLab/Docs/Renderers/`: no
  `LICENSE` file. The gallery scene also pulls two NVIDIA Base materials (Parquet_Floor,
  Marble_Tile_12) from the `Assets/Isaac/2022.2.1/NVIDIA/Materials/` tree, also without a licence file.
- So: download and convert locally for benchmarking only. Never commit, upload or redistribute the
  USD files, the textures or the converted MJCF. `tools/fetch_assets.py --only isaac` prints this
  note and is opt-in (a plain `fetch_assets.py` run does not fetch Isaac assets).

| scene (use) | path under `{root}/Isaac/` | size | materials | lights | to MJCF |
| --- | --- | --- | --- | --- | --- |
| renderer gallery scene (IsaacLab renderer docs) | `IsaacLab/Docs/Renderers/renderer_gallery_scene.usda` | 0.03 MB | 10 UsdPreviewSurface (+2 remote MDL floors) | Dome, Rect | easy with a USD converter |
| Franka-Cabinet (`Isaac-RenderBenchmark-Franka-Cabinet`, Newton camera benchmark) | `IsaacLab/Robots/FrankaEmika/franka_panda.usda` + `Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd` | ~37 MB | OmniPBR / PreviewSurface constants | Dome 2000 + Distant 200 (task config) | easy (constants only) |
| SeattleLabTable (lift / stack tasks) | `Props/Mounts/SeattleLabTable/table.usd` | 77 MB | OmniPBR with diffuse / normal / metallic / roughness textures | – | medium (OmniPBR mapping) |
| Simple_Room (Isaac Sim tutorials) | `Environments/Simple_Room/simple_room.usd` | 117 MB | OmniPBR textures, PreviewSurface, OmniGlass | Rect + Dome | medium |
| replicator_kitchen | `Environments/replicator_kitchen/kitchen_l_island.usda` | 23 MB+ | PreviewSurface + NVIDIA Base MDLs | Dome + 3 sphere + 7 rect + 3 disk | medium-hard; best authored-light layout |
| Simple / full warehouse | `Environments/Simple_Warehouse/{warehouse,full_warehouse}.usd` | 539 MB | Unreal-exported `MI_*.mdl` | Distant + Rect | hard |
| Hospital / Office | `Environments/{Hospital,Office}/*.usd` | 1.1 GB / 741 MB | custom MDL | Office: 119 sphere lights | not now |

**Newton's own assets.**
- The bundled example USDs (`newton/examples/assets`) have no materials.
- Robot examples download `newton-physics/newton-assets` (Apache-2.0). ANYmal-D there is the
  Isaac asset (albedo JPGs); G1 / H1 / shadow_hand are Menagerie-converted, with plain colours.
- Newton's camera sensor uses albedo and textures only.

## What survives Newton's USD import

- **`ModelBuilder.add_usd`** reads UsdPreviewSurface and OmniPBR inputs by name. It keeps base
  colour or a single base-colour texture, opacity, scalar metallic / roughness and UV transforms.
- **Dropped:** normal, ORM, AO and emissive maps; MDL is not evaluated; UsdLux lights are not read.
- **Newton → MuJoCo (`SolverMuJoCo`)** builds meshes from vertices and faces only, with no UVs,
  textures or materials, and skips visual-only geoms by default.
- **MuJoCo's USD decoder** maps only UsdPreviewSurface diffuse / metallic / roughness and no
  lights.

The PBR path is therefore a dedicated USD → MJCF converter (like `tools/gltf_to_mjcf.py`) on
`usd-core`. It would:
1. triangulate meshes, expand UVs and normals, split by GeomSubset, bake transforms and units;
2. map UsdPreviewSurface and OmniPBR textures (diffuse, normal, ORM / roughness / metallic / AO,
   emissive, opacity) to MJCF texture layers;
3. map UsdLux lights: sphere → point with `bulbradius`; rect / disk → spot approximation;
   distant → directional; dome → envmap. Intensities need a unit calibration.
4. take robot kinematics from the Menagerie twin, or from Newton's `save_to_mjcf`, merged by body
   name.

Estimated effort is 2–4 days for static visual scenes. Suggested order: gallery scene and
Franka-Cabinet, then the OmniPBR-textured props, then the light-rich kitchen.

### Status: `tools/usd_to_mjcf.py` (2026-10-22)

Implemented as above, minus robot kinematics: articulations are posed from UsdPhysics joints
(`joint_positions`) and baked into static visual geoms. Benchmark scenes (`benchmarks/scenes.py`,
fetched with `tools/fetch_assets.py --only isaac` into `$MJPBR_ASSET_DIR/mjcf/<Scene>/`):

| scene | source | converted | lights |
| --- | --- | --- | --- |
| `isaac_renderer_gallery` | gallery scene + 2 remote Base MDLs (25 MB) | PreviewSurface spheres (mirror, glass, emissive); floors via the OmniPBR-wrapping MDLs: base colour, normal, ORM, `texture_scale` | RectLight → spot; the DomeLight is authored `invisible`, so it is dropped |
| `isaac_franka_cabinet` | Franka + Sektion cabinet, composed like `Isaac-RenderBenchmark-Franka-Cabinet` (8 MB) | 67 meshes (instance proxies), PreviewSurface constants, ground cuboid; Franka in IsaacLab's default pose (matches MuJoCo FK of the Menagerie panda) | Dome 2000 → skybox, clamped from 1.2 to 1; Distant 200 lux |
| `isaac_seattle_lab_table` | SeattleLabTable (76 MB, 4k maps downscaled to 2k) | OmniPBR: diffuse, normal, roughness and metallic textures (packed ORM), GeomSubsets | none (standard sun / HDRI) |

**Light calibration.** One factor maps USD intensities to the renderer's physical units:
pixel-per-nit of the USD camera's exposure (`exposure:fStop/time/iso/responsivity`, else Kit's
f/5, 1/50 s, ISO 100 → 8e-4) times 1.2 · 2^EV100 at the renderer's default EV100 (≈ 31–34).
DistantLight intensity is lux; sphere / rect / disk radiance × emitting area gives candela.
With this, the gallery's key light and the Franka-Cabinet dome look as bright as in Kit.

**Findings.**
- Kit's uniform dome lights are bright: Franka-Cabinet's 0.75 × 2000 is 1.2 display-relative.
  The real-time tier's IBL is unoccluded, so the robot shows up as a flat white silhouette. The
  path-traced tier (dome as skybox, occluded) looks right.
- The NVIDIA Base material library MDLs are thin `::OmniPBR::OmniPBR(...)` wrappers. Their
  argument lists are parsed, which gives normal and ORM maps that the scene's PreviewSurface
  fallback lacks.
- USD st and OpenGL-convention normal maps work unchanged: UVs are written to the OBJ as they are,
  and a synthetic bump renders lit from the correct side on both axes.
- Not done: replicator_kitchen. It references ~25 SimReady props and vMaterials / Base MDLs across
  the tree; that needs a dependency-driven fetch and probably exceeds the 150 MB budget.

## glTF PBR showcases (Khronos sample assets)

`gltf_to_mjcf.py` ignores all `KHR_*` extensions, texture transforms, `TEXCOORD_1`, vertex colours
and alpha cutoff.

| asset | licence | size | note |
| --- | --- | --- | --- |
| ABeautifulGame | CC-BY-4.0 | 30 MB | works; glass pieces render opaque (transmission) |
| Sponza (Khronos) | Cryengine licence | 53 MB | needs alpha cutoff for foliage; no lights |
| Corset / AntiqueCamera / Lantern | CC0 | 10–18 MB | work now |
| SciFiHelmet, FlightHelmet, DamagedHelmet, BoomBox, WaterBottle | – | – | already fetched |

**What Filament uses:**
- `gltf_viewer` embeds DamagedHelmet, and the web demo uses FlightHelmet.
- `assets/models/monkey` is an OBJ with separate colour / normal / AO / roughness / metallic maps,
  and needs a hand-written MJCF.
- Filament's render tests use the Khronos list above.

## Robotics MJCF

Most MuJoCo Menagerie models use plain rgba with metallic / roughness unset.

| scene | licence | note |
| --- | --- | --- |
| Menagerie `anybotics_anymal_c` | BSD-3 | 19 albedo textures; works now |
| Menagerie `franka_emika_panda` + ManiSkill `table.glb` | Apache-2.0 | table has base / metallic-roughness / normal maps (via gltf_to_mjcf); copy ManiSkill's panda material values |
| Menagerie `aloha` | BSD-3 | table albedo texture |
| MuJoCo Warp `benchmarks/kitchen` | MIT (RoboCasa) | geometry-heavy, rgba only |

## Light-rig scenes (no HDRI)

- **`KHR_lights_punctual` is not converted yet.** Converting it is about half a day:
  - point / spot intensity in candela, directional in lux;
  - spot outer cone → `cutoff`;
  - scale intensity by factor² under `--normalize`.
- **LightsPunctualLamp** (CC-BY-4.0, 5 point lights) and **PointLightIntensityTest** (CC0, 8 point
  lights; doubles as a candela calibration test) need this converter feature.
- **A robotics showroom** works now in MJCF: Franka or ANYmal with an authored key spot, a fill
  point light with `bulbradius` and a rim light.
- Avoid PlaysetLightTest (CC-BY-NC-SA).
