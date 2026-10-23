# Copyright 2026 Julien Blanchon
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

"""OptiX traversal backend: parity with the software traversal (skipped without PyOptiX)."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import optix as optix_backend

_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <light type="directional" dir="0.3 0.2 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2" euler="0 0 25"/>
      <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2" euler="40 0 0"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2" euler="30 0 0"/>
      <geom type="ellipsoid" pos="0.3 0.6 0.3" size="0.1 0.2 0.15"/>
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="mesh" mesh="ico" euler="20 30 10"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="ico" scale="0.2 0.2 0.2"
        vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
    </asset>
  </mujoco>
"""


def _available() -> bool:
  return optix_backend.unavailable_reason(mujoco.MjModel.from_xml_string(_XML), (0, 1, 2), mjpbr.Tier.REALTIME) is None


pytestmark = pytest.mark.skipif(not _available(), reason="OptiX backend unavailable (PyOptiX / SDK headers)")


def _render_both(xml: str, nworld: int, move: bool = False, **kwargs):
  """Renders with both traversals; returns {traversal: (rgb, depth, seg)}."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  if move:
    # Different free-body poses per world, including a large displacement.
    qpos = np.tile(mjd.qpos, (nworld, 1))
    rng = np.random.default_rng(0)
    qpos[:, :3] += rng.normal(0.0, 0.3, (nworld, 3))
    qpos[-1, :3] = [2.5, 1.0, 0.5]
    quat = rng.normal(size=(nworld, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    d.qpos.assign(qpos.astype(np.float32))
    mjw.forward(m, d)
  out = {}
  for traversal in ("software", "optix"):
    rc = mjpbr.create_render_context(mjm, nworld=nworld, traversal=traversal, **kwargs)
    assert rc.traversal == traversal
    mjw.refit_bvh(m, d, rc.base)
    mjpbr.render(m, d, rc)
    out[traversal] = (rc.base.rgb_data.numpy(), rc.base.depth_data.numpy(), rc.base.seg_data.numpy())
  return out


def _rgb8(packed: np.ndarray) -> np.ndarray:
  return packed.view(np.uint8).reshape(*packed.shape, 4)[..., :3].astype(int)


@pytest.mark.parametrize("render_rgb", [True, False])
@pytest.mark.parametrize("move", [False, True])
def test_depth_and_segmentation_match_software(render_rgb, move):
  out = _render_both(
    _XML,
    nworld=3,
    move=move,
    cam_res=(48, 40),
    render_rgb=render_rgb,
    render_depth=True,
    render_seg=True,
  )
  (_, ds, ss), (_, do, so) = out["software"], out["optix"]
  # Grazing hits may flip between geoms at silhouettes: allow a few pixels.
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998
  same = np.all(ss == so, axis=-1)
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("samples_per_pixel", [1, 2])
def test_rgb_close_to_software(samples_per_pixel):
  out = _render_both(
    _XML,
    nworld=2,
    move=True,
    cam_res=(64, 48),
    render_rgb=True,
    render_depth=True,
    render_seg=True,
    samples_per_pixel=samples_per_pixel,
  )
  a, b = _rgb8(out["software"][0]), _rgb8(out["optix"][0])
  diff = np.abs(a - b).max(axis=-1)
  assert a.mean() > 20.0  # something was rendered
  assert np.mean(diff > 2) < 0.005
  assert np.abs(a - b).mean() < 0.1
  # Depth / segmentation of the first sub-pixel sample.
  assert np.mean(np.all(out["software"][2] == out["optix"][2], axis=-1)) > 0.998


def test_transparent_geoms_are_skipped():
  xml = _XML.replace(
    '<geom type="sphere" pos="0 0 0.4" size="0.3"/>', '<geom type="sphere" pos="0 0 0.4" size="0.3" rgba="1 0 0 0"/>'
  )
  out = _render_both(xml, nworld=2, cam_res=(32, 24), render_rgb=False, render_depth=True, render_seg=True)
  ss, so = out["software"][2], out["optix"][2]
  assert not np.any(so[..., 0] == 1)  # geom 1 is the invisible sphere
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998


def test_falls_back_for_height_fields():
  xml = """
    <mujoco>
      <asset><hfield name="h" nrow="4" ncol="4" size="1 1 0.1 0.1"/></asset>
      <worldbody>
        <geom type="hfield" hfield="h"/>
        <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  with pytest.warns(UserWarning, match="falls back"):
    rc = mjpbr.create_render_context(mjm, cam_res=(8, 8), traversal="optix")
  assert rc.traversal == "software"


def _setup(xml: str, nworld: int, **kwargs):
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  rc = mjpbr.create_render_context(mjm, nworld=nworld, traversal="optix", **kwargs)
  assert rc.traversal == "optix"
  return mjm, mjd, m, d, rc


def _move(m, d, mjd, seed: int):
  """Moves the free body differently in every world (the refits must follow)."""
  rng = np.random.default_rng(seed)
  qpos = np.tile(mjd.qpos, (d.nworld, 1))
  qpos[:, :3] += rng.normal(0.0, 0.2, (d.nworld, 3))
  quat = rng.normal(size=(d.nworld, 4))
  qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  d.qpos.assign(qpos.astype(np.float32))
  mjw.forward(m, d)


def _outputs(rc):
  return [a.numpy().copy() for a in (rc.base.rgb_data, rc.base.depth_data, rc.base.seg_data)]


@pytest.mark.parametrize("samples_per_pixel", [1, 2])
def test_graph_capture_matches_eager(samples_per_pixel):
  _, mjd, m, d, rc = _setup(
    _XML,
    nworld=3,
    cam_res=(32, 24),
    render_rgb=True,
    render_depth=True,
    render_seg=True,
    samples_per_pixel=samples_per_pixel,
  )
  rc.optix_graph_capture = True
  mjpbr.render(m, d, rc)  # compiles and builds outside the capture
  with wp.ScopedCapture() as capture:
    mjpbr.render(m, d, rc)
  for seed in range(3):
    _move(m, d, mjd, seed)
    mjpbr.render(m, d, rc)
    eager = _outputs(rc)
    for a in (rc.base.rgb_data, rc.base.depth_data, rc.base.seg_data):
      a.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize()  # before the pageable copies of _outputs (see PbrRenderContext)
    graph = _outputs(rc)
    assert (eager[2][..., 0] == 6).sum() > 20  # the moving mesh is in view
    for a, b in zip(eager, graph):
      np.testing.assert_array_equal(a, b)


def test_graph_capture_needs_opt_in_and_an_eager_render_first():
  _, _, m, d, rc = _setup(_XML, nworld=2, cam_res=(16, 16), render_rgb=True)
  with pytest.raises(RuntimeError, match="optix_graph_capture"):
    with wp.ScopedCapture():
      mjpbr.render(m, d, rc)
  rc.optix_graph_capture = True
  with pytest.raises(RuntimeError, match="outside the CUDA graph capture"):
    with wp.ScopedCapture():
      mjpbr.render(m, d, rc)


def test_primitive_layout():
  """Spheres and capsules are linear curves, boxes unit-cube instances, the rest custom."""
  rng = np.random.default_rng(1)
  geoms = []
  for i in range(24):
    kind = ("sphere", "capsule", "box", "cylinder", "ellipsoid", "capsule")[i % 6]
    size = {"sphere": "0.12", "capsule": "0.06 0.15", "box": "0.1 0.05 0.12", "cylinder": "0.07 0.1"}
    pos = " ".join(f"{x:.3f}" for x in (rng.uniform(-1.2, 1.2), rng.uniform(-0.8, 0.8), rng.uniform(0.15, 0.8)))
    euler = " ".join(f"{x:.1f}" for x in rng.uniform(-90, 90, 3))
    geoms.append(f'<geom type="{kind}" size="{size.get(kind, "0.1 0.06 0.12")}" pos="{pos}" euler="{euler}"/>')
  xml = f"""
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        {"".join(geoms)}
        <camera name="cam" pos="0 -2.8 1.4" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  out = _render_both(xml, nworld=2, cam_res=(64, 48), render_rgb=True, render_depth=True, render_seg=True)
  (rs, ds, ss), (ro, do, so) = out["software"], out["optix"]
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998
  same = np.all(ss == so, axis=-1) & (ss[..., 0] > 0)
  assert same.sum() > 500
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)
  diff = np.abs(_rgb8(rs) - _rgb8(ro)).max(axis=-1)
  assert np.mean(diff > 2) < 0.005

  mjm = mujoco.MjModel.from_xml_string(xml)
  rc = mjpbr.create_render_context(mjm, cam_res=(8, 8), traversal="optix")
  mjpbr.render(mjw.put_model(mjm), mjw.put_data(mjm, mujoco.MjData(mjm)), rc)
  scene = rc._optix.scene
  types = mjm.geom_type
  assert {int(types[g]) for g in scene.curve_geom_list} == {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
  }
  assert {int(types[g]) for g in scene.box_geom_list} == {int(mujoco.mjtGeom.mjGEOM_BOX)}
  assert set(scene.primitive_types) == {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}


def _l_shape_mesh() -> str:
  """An axis-aligned L-shaped prism; MuJoCo's inertia frame is rotated from the authored one."""

  def box(lo, hi, base):
    verts = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    faces = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4)]
    faces += [(2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    return verts, [tuple(i + base for i in f) for f in faces]

  v1, f1 = box((0, 0, 0), (2, 0.3, 0.2), 0)
  v2, f2 = box((0, 0.3, 0), (0.3, 1.2, 0.2), 8)
  vertex = " ".join(f"{c:g}" for v in v1 + v2 for c in v)
  face = " ".join(str(i) for f in f1 + f2 for i in f)
  return f'<mesh name="l" inertia="shell" scale="0.5 0.5 0.5" vertex="{vertex}" face="{face}"/>'


def test_reframed_mesh_gas_matches_software():
  """A triangle GAS built in the authored frame (mesh_bvh_frame="auto") gives the same hits."""
  xml = f"""
    <mujoco>
      <asset>{_l_shape_mesh()}</asset>
      <worldbody>
        <light pos="0.5 -0.5 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <body pos="-0.3 -0.2 0.1" euler="10 0 25">
          <freejoint/>
          <geom type="mesh" mesh="l"/>
        </body>
        <camera name="cam" pos="0 -2.2 1.6" xyaxes="1 0 0 0 0.6 0.8"/>
      </worldbody>
    </mujoco>
  """
  options = dict(cam_res=(48, 40), render_rgb=True, render_depth=True, render_seg=True)
  out = _render_both(xml, nworld=2, move=True, **options)
  ref = _render_both(xml, nworld=2, move=True, mesh_bvh_frame="mujoco", **options)["optix"]
  mjm = mujoco.MjModel.from_xml_string(xml)
  assert mjpbr.create_render_context(mjm, traversal="optix", cam_res=(8, 8)).mesh_bvh_frame is not None
  (_, ds, ss), (ro, do, so) = out["software"], out["optix"]
  assert (so[..., 0] == 1).sum() > 100  # the mesh fills part of the image
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998
  same = np.all(ss == so, axis=-1)
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)
  # Against the GAS in MuJoCo's mesh frame.
  assert np.mean(np.all(ref[2] == so, axis=-1)) > 0.998
  assert np.abs(_rgb8(ref[0]) - _rgb8(ro)).max() <= 2


@pytest.mark.parametrize("translucent", [False, True])
def test_reflections_match_software(translucent):
  """Mirror reflections, traced by the camera-layer loop in the OptiX program, match software."""
  xml = _XML.replace('<geom type="plane" size="3 3 0.1"/>', '<geom type="plane" size="3 3 0.1" material="mirror"/>').replace(
    "<asset>", '<asset><material name="mirror" rgba="0.6 0.6 0.7 1" reflectance="0.4"/>'
  )
  if translucent:
    xml = xml.replace('size="0.3"/>', 'size="0.3" rgba="0.2 0.8 0.2 0.5"/>')
  out = _render_both(xml, nworld=2, move=True, cam_res=(64, 48), render_rgb=True, render_depth=True, render_seg=True)
  a, b = _rgb8(out["software"][0]), _rgb8(out["optix"][0])
  diff = np.abs(a - b).max(axis=-1)
  assert np.mean(diff > 2) < 0.005
  assert np.abs(a - b).mean() < 0.15


def test_auto_traversal():
  """traversal="auto": OptiX when meshes are rendered, software for primitive-only scenes."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  assert mjpbr.create_render_context(mjm, cam_res=(8, 8), traversal="auto").traversal == "optix"
  no_mesh = mujoco.MjModel.from_xml_string(
    _XML.replace('<geom type="mesh" mesh="ico" euler="20 30 10"/>', '<geom type="sphere" size="0.1"/>')
  )
  assert mjpbr.create_render_context(no_mesh, cam_res=(8, 8), traversal="auto").traversal == "software"
  assert mjpbr.create_render_context(mjm, cam_res=(8, 8), traversal="auto", tier=mjpbr.Tier.PATH).traversal == "software"
