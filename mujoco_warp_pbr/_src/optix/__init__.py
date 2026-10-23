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

"""Hardware ray-tracing (OptiX) traversal backend of the real-time tier.

Requirements: Warp >= 1.18 (experimental external-compilation API), PyOptiX
(github.com/NVIDIA/otk-pyoptix) built against the OptiX ABI of the installed driver, and the OptiX
SDK headers (github.com/NVIDIA/optix-dev) found through OPTIX_INCLUDE_DIR.

- `device.py`: OptiX device intrinsics registered as Warp builtins.
- `trace.py`: `trace_closest` / `trace_any` over OptiX, world cells, intersection programs.
- `accel.py`: device context, mesh / primitive GASes and the IAS over all worlds.
- `launch.py`: programs, pipeline, shader binding table and per-frame launch.
- `graph.py`: CUDA graph capture of the OptiX calls through a host-function bridge.

Nothing here imports PyOptiX at module import time.
"""

from typing import TYPE_CHECKING, Iterable, Optional

from mujoco_warp_pbr._src.optix.device import AVAILABLE as AVAILABLE
from mujoco_warp_pbr._src.optix.device import UNAVAILABLE_REASON as UNAVAILABLE_REASON

if TYPE_CHECKING:
  import mujoco

  from mujoco_warp_pbr._src.types import PbrRenderContext


def unavailable_reason(mjm: "mujoco.MjModel", enabled_geom_groups: Iterable[int], tier: int) -> Optional[str]:
  """None if traversal="optix" can render this model, else the reason it cannot."""
  import warp as wp

  from mujoco_warp_pbr._src.types import Tier

  if tier != Tier.REALTIME:
    return "it only applies to the real-time tier"
  if UNAVAILABLE_REASON is not None:
    return UNAVAILABLE_REASON
  if not wp.get_device().is_cuda:
    return "it needs a CUDA device"
  try:
    import optix  # noqa: F401
  except Exception as e:  # ImportError, or a missing CUDA runtime library
    return f"PyOptiX is not available ({e})"

  from mujoco_warp_pbr._src.optix import accel
  from mujoco_warp_pbr._src.optix import device

  if device.include_dir() is None:
    return "the OptiX SDK headers were not found (set OPTIX_INCLUDE_DIR)"
  reason = accel.supported(mjm, enabled_geom_groups)
  if reason is not None:
    return f"the scene has {reason}"
  return None


def check(rc: "PbrRenderContext") -> None:
  """Raises the first error of OptiX work that ran inside a replayed CUDA graph, if any.

  A graph captured with `rc.optix_graph_capture` runs the OptiX calls on a worker thread; their
  errors cannot propagate through `wp.capture_launch`. The next `render(rc)` raises them too, but
  a loop that only replays graphs should call this after synchronising (`wp.synchronize()`).

  Raises:
    RuntimeError: chained to the worker's error.
  """
  from mujoco_warp_pbr._src.optix import graph

  if rc._optix is not None:
    rc._optix.check()
  else:
    graph.raise_released_error()
