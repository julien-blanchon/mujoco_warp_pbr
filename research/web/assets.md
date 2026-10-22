# Test assets for PBR conformance, performance, and RL scenes

Survey date: 2026-10-22. Licenses are as listed on each source page. Re-check per asset
before redistributing or committing anything (keep downloaded assets out of the repository,
in a cache directory).

---

## 1. Khronos glTF-Sample-Assets

- URL: https://github.com/KhronosGroup/glTF-Sample-Assets (list: `Models/Models.md`; tag views `#core`, `#extension`, `#testing`, `#pbrtest`, `#showcase`, `#issues`). Each entry has a viewer link and GLB download.
- Per-model licenses differ (CC0 1.0, CC-BY 4.0, some proprietary). The table below was verified from the list:

| Model | Tests | License |
|---|---|---|
| MetalRoughSpheresNoTextures | metallic × roughness grid, factors only | CC0 |
| MetalRoughSpheres | same, textured | CC-BY 4.0 |
| CompareMetallic / CompareRoughness / CompareBaseColor / CompareNormal / CompareAmbientOcclusion / CompareSpecular / CompareIor | side-by-side single-property comparisons | CC0 |
| EnvironmentTest | metal and dielectric spheres, roughness 0..1, IBL check | Adobe Stock license (do not redistribute) |
| AlphaBlendModeTest | OPAQUE / MASK (cutoff) / BLEND | CC-BY 4.0 |
| NormalTangentTest | normal map with auto-generated tangents (MikkTSpace) | CC0 |
| NormalTangentMirrorTest | normal map with supplied tangents, mirrored UVs | CC-BY 4.0 |
| TextureCoordinateTest | UV orientation and placement | CC0 |
| TextureSettingsTest | single/double-sided, wrap modes | CC-BY 4.0 |
| TextureLinearInterpolationTest | filtering after sRGB decode | CC0 |
| TextureEncodingTest | correct transfer functions (sRGB vs linear) | CC0 |
| TextureTransformTest | KHR_texture_transform | CC0 |
| BoxTextured / BoxTexturedNonPowerOfTwo | basic texturing, NPOT | CC-BY 4.0 |
| BoxVertexColors / VertexColorTest | COLOR_0 | CC0 / CC-BY 4.0 |
| MultiUVTest | TEXCOORD_1 | CC-BY 4.0 |
| NegativeScaleTest | negative-determinant transforms (winding flip) | CC-BY 4.0 |
| OrientationTest | node TRS | CC-BY 4.0 |
| BoomBox | emissive | CC0 |
| WaterBottle | basic textured metal/rough | CC0 |
| DamagedHelmet | classic textured PBR showcase | CC-BY 4.0 (older version CC-BY-NC) |
| FlightHelmet | multi-material, alpha | CC0 |
| SciFiHelmet | dense textured asset | CC0 |
| Suzanne | simple mesh | CC0 |
| Sponza (Crytek) | interior lighting, alpha-masked foliage | Cryengine Limited License |
| SpecGlossVsMetalRough | KHR_materials_pbrSpecularGlossiness | CC-BY 4.0 |
| Extensions (later tiers): ToyCar (transmission/clearcoat/sheen), TransmissionTest, SheenChair, ABeautifulGame (transmission/volume) | | CC0 / CC0 / CC0 / CC-BY 4.0 |

## 2. Poly Haven (HDRIs, textures, models)

- API: `https://api.polyhaven.com` (verified by live query 2026-10-22). Swagger at https://github.com/Poly-Haven/Public-API.
  - `/types` returns `["hdris","textures","models"]`.
  - `/assets?type=hdris&categories=indoor` returned 301 indoor HDRIs, each with metadata such as `evs_cap`, `whitebalance`, `categories`, `tags`, `authors`, and `backplates`.
  - `/files/{id}` returns `{"hdri": {"1k","2k","4k","8k","16k": {"hdr": {url, size, md5}, "exr": {...}}}, "tonemapped": ...}`.
  - URL pattern: `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/<id>_1k.hdr`. For example, kloofendal_48d_partly_cloudy_puresky at 1k is 1.4 MB as .hdr and 5.5 MB as .exr.
- License: **all assets CC0**. API code is AGPL-3.0.
- API ToS: free, including commercial use. Every request **must send a unique User-Agent or Referer** naming your software. Show a "Powered by Poly Haven" credit only if you display content via the live API. No stated rate limits; don't be disruptive. md5 sums are provided, so a verified local cache is easy.
- Good HDRI picks for RL domain randomization: 1k-2k is plenty for 64-512 px observations and IBL prefiltering. Use a mix of indoor/studio (soft) and outdoor sun (hard light, which stresses env importance sampling).

## 3. Intel Sponza (GPU Research Samples)

- URL: https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html
- A 2022 photogrammetry-matched PBR remake of the Sponza atrium: 4K textures, high-resolution geometry, about 2.97 GB base download. Formats: 3ds Max, glTF, FBX, USD. Add-ons include curtains/ivy (unverified), flood, and volumetric explosion.
- License is ambiguous across Intel pages ("Creative Commons Attribution" vs "Academy Software Foundation"; unverified). Check the license in the download.
- Too large for routine CI. Use as an occasional geometry-heavy stress test (millions of tris, many textures) comparable to HSSD in Madrona's benchmark.

## 4. McGuire Computer Graphics Archive

- URL: https://casual-effects.com/data (iframe to https://casual-effects.com/g3d/data10/index.html; verified text)
- OBJ+PNG zips, cleaned by McGuire and collaborators. EGSR 2021 Dataset award. Requested citation: "Morgan McGuire, Computer Graphics Archive, July 2017 (https://casual-effects.com/data)".
- Classic models (the list is loaded by JS and was not read; from prior knowledge, verify per model): Crytek Sponza, San Miguel, Amazon Lumberyard Bistro, Conference Room, Sibenik, Fireplace Room, Breakfast Room, Living Room, Hairball, Buddha, Dragon, Bunny.
- Per-model licenses are listed on the archive, mixed CC-BY, CC0, and others (unverified per model).
- These are the standard traversal benchmark scenes (Conference, Sibenik, Fairy, and San Miguel appear in Aila-Laine and CWBVH), so we can compare Mrays/s directly.

## 5. MuJoCo Menagerie

- URL: https://github.com/google-deepmind/mujoco_menagerie (verified README)
- 11 categories: humanoids (Unitree H1/G1, TALOS, Apptronik Apollo), quadrupeds (Go2, Spot, ANYmal), bipeds (Cassie), biomechanical, dual arms (ALOHA), mobile manipulators (Google Robot, TidyBot, TIAGo), drones, arms (Panda, UR5e, iiwa 14, Kinova Gen3), end-effectors (Shadow, Allegro, Leap, Robotiq 2F-85), mobile bases, misc (RealSense D435i).
- Assets are .stl/.obj meshes for visual and collision. Many models have `<model>_mjx.xml` and `scene_mjx.xml` variants (primitive-heavy collision, suitable for batch sim).
- **Per-model licenses**: BSD-3-Clause, Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause-Clear. Check each `LICENSE`.
- Materials are mostly MuJoCo `rgba`/material colors split across multiple OBJ parts, with few textures (unverified generalization). PBR parameters (metallic, roughness) are not specified, so we need a mapping from MuJoCo `material` (specular, shininess, reflectance, rgba) to glTF metallic-roughness. Current MuJoCo does define `mat_metallic` and `mat_roughness` (nmat×1) plus texture roles (`mjTEXROLE_*`: RGB, normal, ORM, ...), verified in the local header `context/mujofil/third_party/mujoco/include/mujoco/mjmodel.h:699`. Use these when present.

## 6. Embodied-AI scene datasets (for parity with Madrona's benchmark)

- HSSD (Habitat Synthetic Scenes Dataset, https://3dlg-hcvc.github.io/hssd/): about 7.4-7.6M tris and about 330 instances per scene. Used by Madrona as the "complex" benchmark. License terms are on HF and gated (unverified).
- ProcTHOR (about 244K tris per house) and Madrona Hide&Seek (primitives) round out Madrona's benchmark.

---

## Implications for our renderer (ranked by expected impact)

1. **Build a conformance suite from CC0 Khronos models**: MetalRoughSpheresNoTextures, the Compare* set, NormalTangentTest, TextureCoordinateTest, TextureEncodingTest, TextureLinearInterpolationTest, BoxVertexColors, BoomBox, WaterBottle, FlightHelmet, SciFiHelmet. Use them with golden images from the Khronos glTF-Sample-Viewer or Filament (`context/filament`) under the same HDRI and tone mapper (PBR Neutral). Avoid EnvironmentTest (Adobe license) and Sponza (Cryengine) in redistributable test data.
2. **Add the CC-BY tests that catch real bugs**: AlphaBlendModeTest (MASK any-hit), NegativeScaleTest (winding and normal flip), NormalTangentMirrorTest (supplied tangents), TextureSettingsTest (double-sided). Keep attribution in a NOTICE file.
3. **HDRIs: a small pinned Poly Haven set** (about 8-16 maps at 1k-2k, md5-verified, set a User-Agent) cached outside the repo for IBL and env importance sampling, plus 1-2 high-dynamic-range sun maps to stress MIS.
4. **Performance scenes**: (a) Menagerie robots (Panda, Go2, Shadow Hand, ALOHA, Apollo) in `scene_mjx.xml`-style worlds, which is the actual RL workload. (b) McGuire Conference, Sibenik, and San Miguel for Mrays/s comparisons with Aila-Laine and CWBVH tables. (c) Intel Sponza or HSSD occasionally for million-triangle stress.
5. **Material mapping is a spec item**: define how MuJoCo materials (rgba, specular, shininess, reflectance, texture) map to baseColor, metallic, and roughness, and let XML or a sidecar glTF override them. Menagerie mostly lacks PBR data.
6. **Respect licenses in CI**: download at test time into a cache directory outside the repo, store md5s, and gate non-CC0 assets behind an opt-in flag.
