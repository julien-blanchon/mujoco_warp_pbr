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

"""Host side of a megakernel launch: input structs and the specialised-kernel cache."""

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model

from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src.types import PbrRenderContext


@dataclasses.dataclass
class Inputs:
  """Structs of one megakernel launch and the settings values they were built from."""

  geo: trace.Geometry
  mats: material.Materials
  lit: lights.Lights
  cams: kernel_inputs.Cameras
  env: kernel_inputs.Environment
  out: kernel_inputs.Outputs
  values: dict[str, Any]

  @property
  def structs(self) -> list:
    """The leading kernel arguments (geo, mats, lit, cams, env)."""
    return [self.geo, self.mats, self.lit, self.cams, self.env]


def _sources(m: Model, d: Data, rc: PbrRenderContext) -> list:
  """Every object the input structs are built from: the fields of m, d, rc and rc.base."""
  public = [v for k, v in vars(rc).items() if not k.startswith("_")]
  return [*vars(m).values(), *vars(d).values(), *vars(rc.base).values(), *public]


def _warn_ignored_overrides(m: Model, rc: PbrRenderContext):
  """Warns when replaced colours / reflectances need a flag that is off (once per array).

  has_transparency and has_reflection are derived from the model at creation; arrays that
  introduce translucency or reflectance later need them set (see PbrRenderContext). The arrays
  of the first render are the model's own and are not checked.
  """
  device = wp.get_device()
  if device.is_cuda and wp.get_stream(device).is_capturing:
    return
  cache = rc._host_cache
  checks = (
    (
      "geom_rgba",
      m.geom_rgba,
      rc.has_transparency,
      lambda a: np.any((a[..., 3] > 0.0) & (a[..., 3] < 1.0)),
      "has_transparency",
    ),
    ("mat_reflectance", rc.mat_reflectance, rc.has_reflection, lambda a: np.any(a > 0.0), "has_reflection"),
  )
  for name, array, enabled, needs, flag in checks:
    seen = cache.get(name)
    cache[name] = array
    if seen is None or seen is array or enabled:
      continue
    if needs(array.numpy()):
      warnings.warn(f"the replaced {name} is ignored by the renderer: set rc.{flag} = True to render it", stacklevel=4)


def launch_inputs(m: Model, d: Data, rc: PbrRenderContext) -> Inputs:
  """Kernel input structs of the software traversal for the current frame.

  Building the structs costs ~0.1 ms of host time per frame, so they are cached and rebuilt only
  when a field of m, d, rc or rc.base was replaced (e.g. `rc.geom_roughness = ...`).
  """
  kernel_inputs.check_env_id(rc)
  trace.update_trace_frames(m, d, rc)
  sources = _sources(m, d, rc)
  ids = tuple(map(id, sources))
  cached = rc._host_cache.get("inputs")
  # The cache holds the sources themselves, so their ids cannot be reused by new objects.
  if cached is not None and cached[1] == ids:
    return cached[2]
  _warn_ignored_overrides(m, rc)
  values = kernel_inputs.settings_values(rc)
  inputs = Inputs(
    geo=trace.software_geometry(m, d, rc),
    mats=material.materials(m, rc),
    lit=lights.lights(m, d, rc),
    cams=kernel_inputs.cameras(m, d, rc),
    env=kernel_inputs.environment(rc, values),
    out=kernel_inputs.outputs(rc),
    values=values,
  )
  rc._host_cache["inputs"] = (sources, ids, inputs)
  return inputs


def cached_kernel(
  m: Model, rc: PbrRenderContext, values: dict[str, Any], build: Callable[[Model, PbrRenderContext], wp.Kernel]
) -> wp.Kernel:
  """The context's megakernel, (re)built by `build` when anything it is specialised on changed."""
  key = kernel_inputs.kernel_key(m, rc, values)
  if rc._megakernel is None or rc._megakernel_key != key:
    rc._megakernel = build(m, rc)
    rc._megakernel_key = key
  return rc._megakernel
