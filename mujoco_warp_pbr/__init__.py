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

"""Batched physically based renderer for MuJoCo Warp."""

from mujoco_warp_pbr._src.envmap import load_hdr as load_hdr
from mujoco_warp_pbr._src.io import create_render_context as create_render_context
from mujoco_warp_pbr._src.output import get_depth as get_depth
from mujoco_warp_pbr._src.output import get_hdr as get_hdr
from mujoco_warp_pbr._src.output import get_rgb as get_rgb
from mujoco_warp_pbr._src.output import get_segmentation as get_segmentation
from mujoco_warp_pbr._src.render import render as render
from mujoco_warp_pbr._src.types import LightModel as LightModel
from mujoco_warp_pbr._src.types import PbrRenderContext as PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder as RenderOrder
from mujoco_warp_pbr._src.types import Tier as Tier
from mujoco_warp_pbr._src.types import ToneMapper as ToneMapper
from mujoco_warp_pbr._src.types import UnsetMaterial as UnsetMaterial

__all__ = [
  "LightModel",
  "PbrRenderContext",
  "RenderOrder",
  "Tier",
  "ToneMapper",
  "UnsetMaterial",
  "create_render_context",
  "get_depth",
  "get_hdr",
  "get_rgb",
  "get_segmentation",
  "load_hdr",
  "render",
]
