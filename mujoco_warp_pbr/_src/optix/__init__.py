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

"""Hardware ray-tracing (OptiX) traversal backend of the real-time tier.

Requirements: Warp >= 1.18 (experimental external-compilation API), PyOptiX
(github.com/NVIDIA/otk-pyoptix) built against the OptiX ABI of the installed driver, and the OptiX
SDK headers (github.com/NVIDIA/optix-dev) found through OPTIX_INCLUDE_DIR.

- `device.py`: OptiX device intrinsics registered as Warp builtins.
- `trace.py`: `trace_closest` / `trace_any` over OptiX, world cells, intersection programs.
- `accel.py`: device context, mesh / primitive GASes and the IAS over all worlds.
- `launch.py`: programs, pipeline, shader binding table and per-frame launch.

Nothing here imports PyOptiX at module import time.
"""

from typing import Optional


def unavailable_reason(mjm, enabled_geom_groups, tier: int) -> Optional[str]:
  """None if traversal="optix" can render this model, else the reason it cannot."""
  import warp as wp

  from mujoco_warp_pbr._src.types import Tier

  if tier != Tier.REALTIME:
    return "it only applies to the real-time tier"
  if not (hasattr(wp, "build_experimental") and hasattr(wp, "ModuleBuildOptions")):
    return "Warp >= 1.18 is required"
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
