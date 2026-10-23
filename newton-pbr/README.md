# newton-pbr

Physically based camera sensor and viewer for [Newton](https://github.com/newton-physics/newton),
rendered by [mujoco_warp_pbr](https://github.com/julien-blanchon/mujoco_warp_pbr/blob/main/README.md). Install it in a Newton project to render any Newton scene
(built in code, or imported from USD, URDF or MJCF) with PBR shading, image-based lighting and
ray-traced shadows, batched over all worlds.

Both are drop-in replacements: switch with one import or one constructor.

**Differences from Newton** (details in the table below):
- particles, cloth / deformable meshes and Gaussians are not rendered (warned once);
- every world must have the same visible shapes;
- `RenderConfig.enable_global_world=False` and `TextureProjectionMode.TRIPLANAR` are not
  supported (warned once);
- without `create_default_light`, `lights` or `envmap`, a default key and fill light are used
  (Newton's sensor then has ambient light only);
- `color_image` is PBR shaded and tone mapped, `hdr_color_image` is exposed linear radiance, so
  colours differ from Newton's (geometry outputs match: depth, shape index, normal, albedo).

## Camera sensor

```python
from newton_pbr import SensorCamera  # instead of: from newton.sensors import SensorCamera

camera = SensorCamera(model, envmap="studio.hdr")  # Newton's constructor + optional PBR options
rays = SensorCamera.compute_camera_rays_pinhole(640, 480, camera_fov=1.0, device=model.device)
color = camera.allocate_image_color(view_count, 640, 480)
depth = camera.allocate_image_depth(view_count, 640, 480)

camera.update(state, camera_transforms, rays, color_image=color, depth_image=depth)
```

`newton_pbr.SensorCamera` has the constructor and `update()` of `newton.sensors.SensorCamera`
(Newton 1.7): one view per camera transform, `camera_rays` from any of Newton's
`compute_camera_rays_*` helpers (pinhole, USD, OpenCV, OpenCV fisheye, F-theta,
Kannala-Brandt; they are Newton's own), `world_indices` with `WorldRenderFlag`, `ClearData`,
`RenderConfig`, the `allocate_image_*` helpers, `create_default_light`,
`assign_checkerboard_material` and `compute_camera_transforms_usd`. Each view is rendered as its
own mujoco_warp_pbr world posed from its Newton world, so views can share a world (several
cameras, stereo pairs). Unlike Newton's sensor it refits its own BVH: no
`model.bvh_refit_shapes` is needed.

Extra keyword-only constructor options: `lights=[newton_pbr.Light(...)]`, `envmap=`,
`headlight=` and any `mujoco_warp_pbr.create_render_context` option (`tier=Tier.PATH`,
`samples_per_pixel=`, `envmap_intensity=`, `light_model=`, `traversal="optix"`, ...); they take
precedence over the `RenderConfig` mapping. `update()` also takes `seed=` (soft shadows, path
tracing).

| Newton `SensorCamera` | `newton_pbr.SensorCamera` |
| --- | --- |
| `color_image` (packed RGBA) | PBR shading, tone mapped, sRGB (or linear with `output_color_space=LINEAR`); misses: `clear_color`, or the environment map when one is shown |
| `depth_image`, `forward_depth_image` | same (ray distance in units of the ray direction; forward depth includes ray origins) |
| `shape_index_image`, `normal_image` | same (world-space shading normal) |
| `albedo_image` | surface base colour (shape colour × texture), Newton's encoding |
| `hdr_color_image` | exposed linear radiance before tone mapping |
| misses, disabled / invalid views | `ClearData` values, `DISABLE_PRESERVE` / `DISABLE_CLEAR` as Newton |
| lights | `create_default_light` replaces the lights as Newton; without it (and without `lights` / `envmap`) a key and a fill light are used, where Newton has ambient light only |
| `enable_textures`, `enable_shadows`, `enable_ambient_lighting`, `enable_backface_culling`, `enable_fast_math`, `output_color_space`, `max_distance` | mapped |
| `anti_aliasing` (SSAA / MSAA) with `sample_count > 1` | pixel ray = mean of its samples, `ceil(sqrt(sample_count))²` stratified samples interpolated between neighbouring pixel rays; samples without a direction lower the coverage |
| `render_order`, `block_dim` | mapped when changed from Newton's defaults (else mujoco_warp_pbr's tuned defaults) |
| `enable_global_world=False`, `TextureProjectionMode.TRIPLANAR` | not supported (warned once); textures use mesh UVs and MuJoCo's primitive mapping |
| `tile_width` / `tile_height`, Gaussian settings | ignored |
| particles, deformable triangle meshes (cloth, soft bodies), Gaussians | not rendered (warned once); `sync_deformables` is accepted |
| worlds with different shapes | not supported: every world must have the same visible shapes |

Rays are rendered by mujoco_warp_pbr's ray-bundle mode (`mujoco_warp_pbr.set_camera_rays`),
in both tiers and with OptiX traversal. On simple scenes depth, forward depth, shape index,
normal and albedo match Newton's sensor to rounding (`tests/test_camera.py`, which also runs
Newton's own `SensorCamera` scenarios against this one).

## Viewer

```python
viewer = newton_pbr.ViewerPbr()  # instead of: newton.viewer.ViewerGL()
viewer.set_model(model)
while viewer.is_running():
  viewer.begin_frame(sim_time)
  viewer.log_state(state)
  viewer.end_frame()
```

`ViewerPbr` subclasses `ViewerGL` (same constructor, plus `lights`, `envmap`, `headlight`,
`render_scale` and render options). Every frame it renders all worlds, at the viewer's world
offsets, from the viewer's current camera (pose, field of view, framebuffer size ×
`render_scale`) and shows the image through ViewerGL's fullscreen image path. The window, ImGui
panels, camera controls, picking and pausing are ViewerGL's.

Still drawn by GL only: debug overlays (contacts, joints, centres of mass, logged lines, points
and meshes, gizmos), particles and deformables. Untick "PBR (mujoco_warp_pbr)" in the Rendering
panel, or set `viewer.pbr_enabled = False`, to see them. `set_visible_worlds` only affects the
GL view.

`ViewerPbr(headless=True, width=..., height=..., num_frames=...)` creates no window and no
OpenGL context (no display or GL libraries needed): `get_frame()` returns the PBR frame of the
last `end_frame()` as a `(height, width, 3)` uint8 Warp array, e.g. to record videos.

## Simple sensor

`newton_pbr.SensorPbrCamera` is the original, simpler API: a pinhole image size and field of
view fixed at construction and one view per world.

```python
camera = newton_pbr.SensorPbrCamera(model, width=128, height=128, fovy=60.0, envmap="studio.hdr")
color = camera.allocate_image_color()  # (world_count, height, width), packed RGBA
camera.update(state, camera_transforms, color_image=color)
```

## Scene conversion

**How it works.** The visible shapes of one world (plus global shapes such as the ground; every
world for the viewer) become a MuJoCo model. Each frame, a kernel writes their world poses from
Newton's body transforms into MuJoCo Warp's data; the BVH is refitted and the frame rendered.
MuJoCo kinematics are never used.

**Materials** come from the Newton model:
- shape colours (sRGB) and opacity;
- mesh base-colour textures with UVs;
- scalar mesh metallic / roughness, as imported from USD.

Newton drops normal, ORM and emissive maps at import, so they are not available here.

**Limits of Newton's importers.** For MJCF loaded with `add_mjcf`:
- materials collapse to a colour and scalars, and textures are kept only when the UVs map one
  per vertex;
- lights are dropped;
- planes are collision-only.

`newton_pbr.lights_from_mujoco(mjm, mjd)` and the `headlight=` option restore an MJCF scene's
lighting. Newton shape colours are sRGB, while mujoco_warp_pbr reads MuJoCo `rgba` as linear, so the
same MJCF looks darker through Newton.

**Lights.** Newton models carry no lights. Pass `envmap=` for image-based lighting and / or
`lights=[newton_pbr.Light(...)]`. Without either, a key and a fill directional light are used.

Supported shapes: planes, spheres, capsules, ellipsoids, cylinders, boxes, meshes, convex meshes
and cones (meshed). Height fields and Gaussians are not rendered yet (a warning is printed).
