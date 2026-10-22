# Copyright 2026 The mujoco_warp_pbr Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Real-time PBR megakernel: primary ray, shadow rays, split-sum ambient, tone mapping."""

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.render import vertex_normal
from mujoco_warp._src.render_util import compute_ray
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.warp_util import event_scope

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src.color import make_display_transform
from mujoco_warp_pbr._src.material import Surface
from mujoco_warp_pbr._src.trace import make_trace
from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Shadow-ray origin offset along the surface normal (as in MuJoCo Warp).
_SHADOW_OFFSET = 1.0e-4


def _build_megakernel(m: Model, rc: PbrRenderContext) -> wp.Kernel:
  """Specialises the megakernel on the scene and context configuration."""
  base = rc.base
  trace_closest, trace_any = make_trace(base.geom_ray_types)
  eval_light = lights.make_eval_light(rc.light_model, base.has_spot_lights)
  display_transform = make_display_transform(rc.tone_mapper)

  nlight = m.nlight
  use_shadows = rc.use_shadows
  shadow_light_fraction = rc.shadow_light_fraction
  headlight_active = base.headlight_active
  headlight_diffuse = base.headlight_diffuse
  headlight_specular = base.headlight_specular
  headlight_intensity = rc.headlight_intensity
  znear = base.znear
  zfar = base.zfar
  cull_backfaces = base.enable_backface_culling
  vertex_normals = base.enable_vertex_normals
  render_hdr = rc.render_hdr

  @wp.func
  def shade_light(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_radius: wp.array[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    bvh_id: wp.uint64,
    group_root: int,
    worldid: int,
    bvh_ngeom: int,
    bvh_nflexgeom: int,
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    hfield_bvh_id: wp.array[wp.uint64],
    flex_rgba: wp.array[wp.vec4],
    flex_geom_flexid: wp.array[int],
    flex_geom_edgeid: wp.array[int],
    flex_bvh_id: wp.array[wp.uint64],
    flex_group_root: wp.array2d[int],
    light_type: int,
    light_castshadow: bool,
    light_pos: wp.vec3,
    light_dir: wp.vec3,
    light_diffuse: wp.vec3,
    light_specular: wp.vec3,
    light_attenuation: wp.vec3,
    light_cutoff: float,
    light_exponent: float,
    light_intensity: float,
    light_range: float,
    surface: Surface,
    diffuse_color: wp.vec3,
    f0: wp.vec3,
    alpha: float,
    energy_comp: wp.vec3,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> wp.vec3:
    l, dist, diffuse_scale, specular_scale = eval_light(
      light_type,
      light_pos,
      light_dir,
      light_diffuse,
      light_specular,
      light_attenuation,
      light_cutoff,
      light_exponent,
      light_intensity,
      light_range,
      point,
    )
    nol = wp.dot(normal, l)
    if nol <= 0.0 or (wp.length_sq(diffuse_scale) == 0.0 and wp.length_sq(specular_scale) == 0.0):
      return wp.vec3(0.0)

    visibility = float(1.0)
    if wp.static(use_shadows):
      if light_castshadow:
        max_t = float(1.0e8)
        if light_type != wp.static(lights.LIGHT_DIRECTIONAL):
          max_t = dist - 1.0e-3
        hit_geom, _hit_dist, _hit_n, _hit_u, _hit_v, _hit_f, _hit_mesh = trace_any(
          geom_type,
          geom_dataid,
          geom_matid,
          geom_size,
          geom_rgba,
          flex_vertadr,
          flex_edge,
          flex_radius,
          mat_rgba,
          geom_xpos_in,
          geom_xmat_in,
          flexvert_xpos_in,
          bvh_id,
          group_root,
          worldid,
          bvh_ngeom,
          bvh_nflexgeom,
          enabled_geom_ids,
          mesh_bvh_id,
          hfield_bvh_id,
          flex_rgba,
          flex_geom_flexid,
          flex_geom_edgeid,
          flex_bvh_id,
          flex_group_root,
          point + normal * wp.static(_SHADOW_OFFSET),
          l,
          max_t,
          wp.static(cull_backfaces),
        )
        if hit_geom != -1:
          visibility = wp.static(shadow_light_fraction)

    h = wp.normalize(l + view)
    noh = wp.max(wp.dot(normal, h), 0.0)
    loh = wp.max(wp.dot(l, h), 0.0)
    fr = wp.cw_mul(brdf.specular_lobe(f0, alpha, noh, nov, nol, loh), energy_comp)
    fd = diffuse_color * brdf.diffuse_lambert()
    return (wp.cw_mul(fd, diffuse_scale) + wp.cw_mul(fr, specular_scale)) * (nol * visibility)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"fast_math": base.use_fast_math})
  def render_megakernel(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    cam_projection: wp.array[int],
    cam_fovy: wp.array2d[float],
    cam_sensorsize: wp.array[wp.vec2],
    cam_intrinsic: wp.array2d[wp.vec4],
    light_type: wp.array2d[int],
    light_castshadow: wp.array2d[bool],
    light_active: wp.array2d[bool],
    light_attenuation: wp.array2d[wp.vec3],
    light_cutoff: wp.array2d[float],
    light_exponent: wp.array2d[float],
    light_diffuse: wp.array2d[wp.vec3],
    light_specular: wp.array2d[wp.vec3],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_radius: wp.array[float],
    mesh_faceadr: wp.array[int],
    mesh_normaladr: wp.array[int],
    mesh_normal: wp.array[wp.vec3],
    mat_emission: wp.array2d[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    cam_xpos_in: wp.array2d[wp.vec3],
    cam_xmat_in: wp.array2d[wp.mat33],
    light_xpos_in: wp.array2d[wp.vec3],
    light_xdir_in: wp.array2d[wp.vec3],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    nrender: int,
    bvh_ngeom: int,
    bvh_nflexgeom: int,
    cam_res: wp.array[wp.vec2i],
    cam_id_map: wp.array[int],
    rgb_adr: wp.array[int],
    depth_adr: wp.array[int],
    seg_adr: wp.array[int],
    render_rgb: wp.array[bool],
    render_depth: wp.array[bool],
    render_seg: wp.array[bool],
    bvh_id: wp.uint64,
    group_root: wp.array[int],
    flex_bvh_id: wp.array[wp.uint64],
    flex_group_root: wp.array2d[int],
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    mesh_facenormal: wp.array[wp.vec3i],
    hfield_bvh_id: wp.array[wp.uint64],
    flex_rgba: wp.array[wp.vec4],
    flex_geom_flexid: wp.array[int],
    flex_geom_edgeid: wp.array[int],
    background_color: wp.uint32,
    geom_metallic: wp.array2d[float],
    geom_roughness: wp.array2d[float],
    geom_reflectance: wp.array2d[float],
    light_intensity: wp.array2d[float],
    light_range: wp.array2d[float],
    exposure: wp.array[float],
    ambient_radiance: wp.array[wp.vec3],
    dfg_lut: wp.Texture2D,
    # Out:
    rgb_out: wp.array2d[wp.uint32],
    depth_out: wp.array2d[float],
    seg_out: wp.array2d[wp.vec2i],
    hdr_out: wp.array2d[wp.vec3h],
  ):
    worldid, rayid = wp.tid()

    # Global ray id -> (active camera, pixel) using cumulative image sizes.
    camid = int(-1)
    pixelid = int(-1)
    offset = int(0)
    for i in range(nrender):
      npix = cam_res[i][0] * cam_res[i][1]
      if rayid < offset + npix:
        camid = i
        pixelid = rayid - offset
        break
      offset += npix
    if camid == -1:
      return

    want_rgb = render_rgb[camid]
    if not want_rgb and not render_depth[camid] and not render_seg[camid]:
      return

    cam = cam_id_map[camid]
    img_w = cam_res[camid][0]
    img_h = cam_res[camid][1]
    ray_dir_cam, ray_offset_cam = compute_ray(
      cam_projection[cam],
      cam_fovy[worldid % cam_fovy.shape[0], cam],
      cam_sensorsize[cam],
      cam_intrinsic[worldid % cam_intrinsic.shape[0], cam],
      img_w,
      img_h,
      pixelid % img_w,
      pixelid // img_w,
      wp.static(znear),
    )
    cam_mat = cam_xmat_in[worldid, cam]
    origin = cam_xpos_in[worldid, cam] + cam_mat @ ray_offset_cam
    direction = cam_mat @ ray_dir_cam

    max_dist = float(MJ_MAXVAL)
    if wp.static(zfar > 0.0):
      max_dist = wp.static(zfar) / wp.max(-ray_dir_cam[2], 1.0e-6)

    root = group_root[worldid]
    geom_id, dist, normal, bary_u, bary_v, face_id, mesh_id = trace_closest(
      geom_type,
      geom_dataid,
      geom_matid,
      geom_size,
      geom_rgba,
      flex_vertadr,
      flex_edge,
      flex_radius,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      flexvert_xpos_in,
      bvh_id,
      root,
      worldid,
      bvh_ngeom,
      bvh_nflexgeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      flex_rgba,
      flex_geom_flexid,
      flex_geom_edgeid,
      flex_bvh_id,
      flex_group_root,
      origin,
      direction,
      max_dist,
      wp.static(cull_backfaces),
    )

    if render_seg[camid] and geom_id != -1:
      if geom_id == -2:
        seg_out[worldid, seg_adr[camid] + pixelid] = wp.vec2i(mesh_id, int(ObjType.FLEX))
      else:
        seg_out[worldid, seg_adr[camid] + pixelid] = wp.vec2i(geom_id, int(ObjType.GEOM))

    if render_depth[camid]:
      depth = float(0.0)
      if geom_id != -1:
        # Planar depth: distance projected onto the optical axis (-z in camera frame).
        depth = dist * -ray_dir_cam[2]
      depth_out[worldid, depth_adr[camid] + pixelid] = depth

    if not want_rgb:
      return
    adr = rgb_adr[camid] + pixelid

    if geom_id == -1:
      rgb_out[worldid, adr] = background_color
      if wp.static(render_hdr):
        hdr_out[worldid, adr] = wp.vec3h(wp.float16(0.0))
      return

    # Shading normal: interpolated mesh normals, facing the viewer.
    if wp.static(vertex_normals):
      if geom_id >= 0 and mesh_id >= 0 and face_id >= 0:
        gtype = geom_type[geom_id]
        if gtype == int(GeomType.MESH.value) or gtype == int(GeomType.SDF.value):
          gmat = geom_xmat_in[worldid, geom_id]
          face = wp.transpose(gmat) @ normal
          tri = mesh_facenormal[mesh_faceadr[mesh_id] + face_id]
          nadr = mesh_normaladr[mesh_id]
          vn = (
            vertex_normal(mesh_normal[nadr + tri[0]], face) * bary_u
            + vertex_normal(mesh_normal[nadr + tri[1]], face) * bary_v
            + vertex_normal(mesh_normal[nadr + tri[2]], face) * (1.0 - bary_u - bary_v)
          )
          normal = wp.normalize(gmat @ vn)
    if wp.dot(normal, direction) > 0.0:
      normal = -normal

    point = origin + direction * dist
    view = -direction

    # Surface parameters.
    surface = Surface()
    surface.occlusion = 1.0
    surface.emissive = wp.vec3(0.0)
    if geom_id == -2:
      rgba = flex_rgba[mesh_id]
      surface.metallic = 0.0
      surface.perceptual_roughness = 0.5
      surface.reflectance = 0.5
    else:
      rgba, mat_id = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, geom_id)
      surface.metallic = geom_metallic[worldid % geom_metallic.shape[0], geom_id]
      surface.perceptual_roughness = geom_roughness[worldid % geom_roughness.shape[0], geom_id]
      surface.reflectance = geom_reflectance[worldid % geom_reflectance.shape[0], geom_id]
      if mat_id >= 0:
        emission = mat_emission[worldid % mat_emission.shape[0], mat_id]
        surface.emissive = wp.vec3(rgba[0], rgba[1], rgba[2]) * emission
    surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
    surface.alpha = rgba[3]

    pr, alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
    diffuse_color = surface.base_color * (1.0 - surface.metallic)
    f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    dfg = ibl.sample_dfg(dfg_lut, nov, pr)
    energy_comp = brdf.energy_compensation(f0, dfg)

    # Ambient: constant-radiance environment through the split-sum terms.
    spec_albedo = brdf.specular_albedo(f0, dfg)
    ambient = ambient_radiance[worldid % ambient_radiance.shape[0]]
    radiance = (
      wp.cw_mul(
        ambient,
        wp.cw_mul(diffuse_color, wp.vec3(1.0) - spec_albedo) + wp.cw_mul(spec_albedo, energy_comp),
      )
      * surface.occlusion
    )

    light_world = worldid % light_type.shape[0]
    for li in range(wp.static(nlight)):
      if light_active[worldid % light_active.shape[0], li]:
        radiance += shade_light(
          geom_type,
          geom_dataid,
          geom_matid,
          geom_size,
          geom_rgba,
          flex_vertadr,
          flex_edge,
          flex_radius,
          mat_rgba,
          geom_xpos_in,
          geom_xmat_in,
          flexvert_xpos_in,
          bvh_id,
          root,
          worldid,
          bvh_ngeom,
          bvh_nflexgeom,
          enabled_geom_ids,
          mesh_bvh_id,
          hfield_bvh_id,
          flex_rgba,
          flex_geom_flexid,
          flex_geom_edgeid,
          flex_bvh_id,
          flex_group_root,
          light_type[light_world, li],
          light_castshadow[worldid % light_castshadow.shape[0], li],
          light_xpos_in[worldid, li],
          light_xdir_in[worldid, li],
          light_diffuse[worldid % light_diffuse.shape[0], li],
          light_specular[worldid % light_specular.shape[0], li],
          light_attenuation[worldid % light_attenuation.shape[0], li],
          light_cutoff[worldid % light_cutoff.shape[0], li],
          light_exponent[worldid % light_exponent.shape[0], li],
          light_intensity[worldid % light_intensity.shape[0], li],
          light_range[worldid % light_range.shape[0], li],
          surface,
          diffuse_color,
          f0,
          alpha,
          energy_comp,
          point,
          normal,
          view,
          nov,
        )

    if wp.static(headlight_active):
      radiance += shade_light(
        geom_type,
        geom_dataid,
        geom_matid,
        geom_size,
        geom_rgba,
        flex_vertadr,
        flex_edge,
        flex_radius,
        mat_rgba,
        geom_xpos_in,
        geom_xmat_in,
        flexvert_xpos_in,
        bvh_id,
        root,
        worldid,
        bvh_ngeom,
        bvh_nflexgeom,
        enabled_geom_ids,
        mesh_bvh_id,
        hfield_bvh_id,
        flex_rgba,
        flex_geom_flexid,
        flex_geom_edgeid,
        flex_bvh_id,
        flex_group_root,
        wp.static(lights.LIGHT_DIRECTIONAL),
        False,
        origin,
        -cam_mat[:, 2],
        wp.static(headlight_diffuse),
        wp.static(headlight_specular),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        0.0,
        wp.static(headlight_intensity),
        1.0,
        surface,
        diffuse_color,
        f0,
        alpha,
        energy_comp,
        point,
        normal,
        view,
        nov,
      )

    # Emission is display-relative: emission 1 shows the base colour at full brightness.
    exposure_world = exposure[worldid % exposure.shape[0]]
    radiance += surface.emissive / exposure_world

    if wp.static(render_hdr):
      hdr_out[worldid, adr] = wp.vec3h(radiance)
    display = display_transform(radiance * exposure_world)
    rgb_out[worldid, adr] = pack_rgba_to_uint32(
      display[0] * 255.0 + 0.5, display[1] * 255.0 + 0.5, display[2] * 255.0 + 0.5, 255.0
    )

  return render_megakernel


@event_scope
def render(m: Model, d: Data, rc: PbrRenderContext):
  """Renders the current frame into the context's output buffers.

  Call `mujoco_warp.refit_bvh(m, d, rc.base)` first whenever geometry moved.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The PBR render context on device.
  """
  base = rc.base
  base.seg_data.fill_(wp.vec2i(-1, -1))
  if rc._megakernel is None:
    rc._megakernel = _build_megakernel(m, rc)

  wp.launch(
    kernel=rc._megakernel,
    dim=(d.nworld, base.total_rays),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.cam_projection,
      m.cam_fovy,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.light_type,
      m.light_castshadow,
      m.light_active,
      m.light_attenuation,
      m.light_cutoff,
      m.light_exponent,
      m.light_diffuse,
      m.light_specular,
      m.flex_vertadr,
      m.flex_edge,
      m.flex_radius,
      m.mesh_faceadr,
      m.mesh_normaladr,
      m.mesh_normal,
      m.mat_emission,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.flexvert_xpos,
      base.nrender,
      base.bvh_ngeom,
      base.bvh_nflexgeom,
      base.cam_res,
      base.cam_id_map,
      base.rgb_adr,
      base.depth_adr,
      base.seg_adr,
      base.render_rgb,
      base.render_depth,
      base.render_seg,
      base.bvh_id,
      base.group_root,
      base.flex_bvh_id,
      base.flex_group_root,
      base.enabled_geom_ids,
      base.mesh_bvh_id,
      base.mesh_facenormal,
      base.hfield_bvh_id,
      base.flex_rgba,
      base.flex_geom_flexid,
      base.flex_geom_edgeid,
      base.background_color,
      rc.geom_metallic,
      rc.geom_roughness,
      rc.geom_reflectance,
      rc.light_intensity,
      rc.light_range,
      rc.exposure,
      rc.ambient_radiance,
      rc.dfg_lut,
    ],
    outputs=[base.rgb_data, base.depth_data, base.seg_data, rc.hdr_data],
    block_dim=m.block_dim.render,
  )
