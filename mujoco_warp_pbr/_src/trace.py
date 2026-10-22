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

"""Ray queries used by the renderer.

`make_trace` returns the closest-hit and any-hit funcs the megakernel calls. The current
backend is MuJoCo Warp's scene BVH + per-geom intersection (`cast_ray`), specialised on the
geom types present in the scene. Other backends (own acceleration structures, hardware ray
tracing) only need to provide funcs with the same signature.
"""

from typing import Tuple

import warp as wp
from mujoco_warp._src.render import _make_cast_ray


def make_trace(geom_ray_types: Tuple[int, ...]) -> Tuple[wp.Function, wp.Function]:
  """Returns (trace_closest, trace_any) specialised on the geom types present.

  Both have MuJoCo Warp's `cast_ray` signature and return
  (geom_id, dist, normal, bary_u, bary_v, face_id, mesh_id); geom_id is -1 on a miss and -2 for
  flex hits (mesh_id then holds the flex id). trace_any returns on the first hit within max_dist.
  """
  return _make_cast_ray(geom_ray_types, first_hit=False), _make_cast_ray(geom_ray_types, first_hit=True)
