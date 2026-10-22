# newton-pbr

Physically based camera sensor for [Newton](https://github.com/newton-physics/newton), rendered by
[mujoco_warp_pbr](../README.md). Install it in a Newton project to render any Newton scene
(built in code, or imported from USD, URDF or MJCF) with PBR shading, image-based lighting and
ray-traced shadows, batched over all worlds.

```python
import newton
import warp as wp

import newton_pbr

model = builder.finalize()  # any Newton model; worlds must have the same visible shapes
state = model.state()

camera = newton_pbr.SensorPbrCamera(model, width=128, height=128, fovy=60.0, envmap="studio.hdr")
color = camera.allocate_image_color()  # (world_count, height, width), packed RGBA
depth = camera.allocate_image_depth()
camera_transforms = wp.array([...], dtype=wp.transformf)  # one per world, -Z forward, +Y up

camera.update(state, camera_transforms, color_image=color, depth_image=depth)
```

`update` mirrors `newton.sensors.SensorCamera.update`: one view per world, the same camera
convention and the same output formats:
- `color_image`: packed RGBA `uint32`;
- `depth_image`: ray distance;
- `forward_depth_image`: planar depth;
- `shape_index_image`: shape index;
- `hdr_color_image`: linear HDR.

The pinhole image size and field of view are fixed at construction, instead of being passed as a
ray bundle.

**How it works.** At construction the visible shapes of one world (plus global shapes such as the
ground) become a MuJoCo model. Each frame, a kernel writes their world poses from Newton's body
transforms into MuJoCo Warp's data; the BVH is refitted and the frame rendered. MuJoCo kinematics
are never used.

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

Any other `mujoco_warp_pbr.create_render_context` option passes through as a keyword, e.g.
`tier=mujoco_warp_pbr.Tier.PATH`, `samples_per_pixel=2` or `light_model=...`.

Supported shapes: planes, spheres, capsules, ellipsoids, cylinders, boxes, meshes, convex meshes
and cones (meshed). Height fields and Gaussians are not rendered yet (a warning is printed).
