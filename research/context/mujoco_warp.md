# mujoco_warp renderer (google-deepmind/mujoco_warp @ 6c33934)

Files: `mujoco_warp/_src/render.py` (~1600 lines), `render_util.py`, `bvh.py`, `ray.py`,
`types.py` (`RenderContext`), benchmarks in `benchmarks/{render,aloha,cloth,unitree_g1}`.

## Architecture
- `render(m, d, rc)`: one specialised megakernel `_render_megakernel`, launch dim
  `(nworld, total_rays)`, one thread per (world, pixel). Global ray id → (camera, local id)
  by a cumulative search over `cam_res` (multi-camera, per-camera resolution).
- Kernel built once per context (`rc._megakernel`) with `module="unique"`,
  `enable_backward=False`, `grid_stride=False`, optional `fast_math`.
- Heavy `wp.static` specialisation: `_make_cast_ray(geom_ray_types)` and
  `_make_sample_texture(geom_ray_types)` drop absent geom types; light loop unrolled
  (`range(wp.static(M_NLIGHT))`); feature flags (`use_textures`, `enable_specular`,
  `has_spot_lights`, `light_attenuation_is_default`, `has_orthographic_camera`, ...) compiled
  in/out. Rationale stated in docstrings: register pressure.
- Acceleration: one scene `wp.Bvh` (SAH) over all worlds' geom AABBs with `groups`, per-world
  `group_root`; traversal via `wp.bvh_query_ray` / `bvh_query_next` (unordered, shared-memory
  stack — see `warp.md`). Meshes / hfields: per-mesh `wp.Mesh` BVH (`ray_mesh_with_bvh`, any-hit
  variant for shadows). Flex: per-flex `wp.Mesh`; splats: separate BVH, sorted 32-hit batches.
- Refit per frame: `refit_bvh(m, d, rc)` (scene bounds from geom poses).
- Closest-hit tie break: higher geom id wins on coincident hits (matches GL depth test).
- Optional precomputed per-pixel rays (`use_precomputed_rays`).

## Shading (MuJoCo GL look, not PBR)
- Base color = geom rgba (or material rgba if geom rgba is default 0.5 grey) × RGB-role texture.
- Lambert diffuse + Blinn-Phong specular (`mat_specular`, `mat_shininess*128`), scalar
  emission × base color, headlight ambient/diffuse/specular, per-light ambient.
- Lights: directional / point / spot (cutoff + exponent), GL attenuation polynomial,
  `castshadow` hard shadows via any-hit cast (origin offset 1e-4 along normal);
  `shadow_light_fraction` for occluded contribution.
- Vertex-normal interpolation for meshes (rejects normals >~37° off the face normal).
- Skybox for missed rays (cube faces stacked vertically, MuJoCo's axis swizzle).
- Output: clamp to [0,1] → packed uint32 RGBA (**no sRGB encode, no HDR, no tone mapping**).
- AA: `samples_per_pixel²` *separate launches* into a float accumulation buffer + resolve.
- Depth: planar (distance × cos to optical axis); seg: (objid, objtype) vec2i.

## What mjwarp drops that PBR needs
- `Model` has no `mat_metallic`, `mat_roughness`, `mat_reflectance`; renderer reads only texture
  role RGB (`mat_texid[..., 1]`) although `mat_texid` keeps all `mjNTEXROLE = 10` roles.
- Lights: no `intensity`, `range`, `bulbradius` (to verify against the installed MuJoCo).

## Conventions to follow (CONTRIBUTING.md)
- 2-space indent, 128 cols, Apache header, kernel params `_in`/`_out` in Model/Data order,
  hoist global reads, no launch guards, quaternions (w,x,y,z), CUDA-graph friendly (no syncs).
- Tests: `test_data.fixture(xml=...)` inline, `nworld: 1, 2` heterogeneous worlds, rendering
  tests assert analytic invariants rather than pixel snapshots.
- Benchmarks: `benchmarks/<name>/__init__.py` with `BENCHMARKS`/`ASSETS`, `"function": "render"`,
  `uv run python3 benchmarks/run.py -f <name>`; `mjwarp-testspeed --function=render`.

## Takeaways for us
- Keep: specialisation-by-`wp.static`, per-context kernel cache, 2-level BVH, analytic
  primitives, mjwarp API shape.
- Improve: ordered/near-first scene traversal (bvh_query is unordered with smem stack),
  in-kernel supersampling, HDR + tone mapping + sRGB, PBR material data, IBL.
