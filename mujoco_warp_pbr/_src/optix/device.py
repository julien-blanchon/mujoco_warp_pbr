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

"""OptiX device intrinsics exposed to Warp kernels.

`optixTrace` is variadic and takes its payload registers by reference, so it is wrapped in fixed
payload layouts and registered with Warp's experimental external-compilation API (Warp >= 1.18)
together with the launch-index, ray and hit accessors the programs need. The glue is a CUDA
preamble that includes `<optix_device.h>`; `build_options` adds the OptiX SDK include directory.

Payload layout of a closest-hit trace (6 registers):
  p0: hit distance (float bits), p1: instance id (0xffffffff on a miss), p2: primitive index,
  p3..p5: triangle barycentrics (b1, b2), or the world-space normal of a curve or custom
  primitive. For those p1 holds the hit's world · ngeom + geom code instead: set by the curve
  closest-hit program (`set_hit`), or by a custom intersection program as attribute 3.
An occlusion trace uses one register: 1 if anything is hit, 0 after the miss program.
"""

import ctypes
import os
from pathlib import Path
from typing import Optional

import warp as wp

NUM_PAYLOAD = 6
NUM_ATTRIBUTE = 4

_PREAMBLE = r"""
// Warp's NVRTC compile has no CUDA toolkit include path: include only the device API (optix.h
// pulls optix_host.h -> <cuda.h>) and skip cooperative vectors (they include <cuda_fp16.h>).
#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 0
#include <optix_device.h>

namespace mjpbr_optix {

struct TraceResult {
  float t;
  int instance;
  int primitive;
  float a;
  float b;
  float c;
};

__device__ inline int launch_index() { return int(optixGetLaunchIndex().x); }

__device__ inline TraceResult trace_closest(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                            wp::uint32 ray_flags) {
  unsigned int p0 = 0u, p1 = 0xffffffffu, p2 = 0u, p3 = 0u, p4 = 0u, p5 = 0u;
  optixTrace(OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]), tmin,
             tmax, 0.0f, OptixVisibilityMask(255), ray_flags | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0u, 1u, 0u, p0, p1, p2,
             p3, p4, p5);
  TraceResult r;
  r.t = __uint_as_float(p0);
  r.instance = int(p1);
  r.primitive = int(p2);
  r.a = __uint_as_float(p3);
  r.b = __uint_as_float(p4);
  r.c = __uint_as_float(p5);
  return r;
}

__device__ inline bool trace_any(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax) {
  unsigned int p0 = 1u;  // the miss program clears it
  optixTrace(OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]), tmin,
             tmax, 0.0f, OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             0u, 1u, 0u, p0);
  return p0 != 0u;
}

__device__ inline void miss() { optixSetPayload_0(0u); }

__device__ inline void closesthit_triangle() {
  const float2 b = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetInstanceId());
  optixSetPayload_2(optixGetPrimitiveIndex());
  optixSetPayload_3(__float_as_uint(b.x));
  optixSetPayload_4(__float_as_uint(b.y));
}

__device__ inline void closesthit_custom() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetAttribute_3());
  optixSetPayload_2(optixGetPrimitiveIndex());
  optixSetPayload_3(optixGetAttribute_0());
  optixSetPayload_4(optixGetAttribute_1());
  optixSetPayload_5(optixGetAttribute_2());
}

__device__ inline int instance_id() { return int(optixGetInstanceId()); }
__device__ inline float ray_tmin() { return optixGetRayTmin(); }
__device__ inline float ray_tmax() { return optixGetRayTmax(); }

__device__ inline int primitive_index() { return int(optixGetPrimitiveIndex()); }

__device__ inline wp::vec3 world_ray_origin() {
  const float3 v = optixGetWorldRayOrigin();
  return wp::vec3(v.x, v.y, v.z);
}

__device__ inline wp::vec3 world_ray_direction() {
  const float3 v = optixGetWorldRayDirection();
  return wp::vec3(v.x, v.y, v.z);
}

// Closest-hit payload of a curve hit: distance, world · ngeom + geom, primitive, world normal.
__device__ inline void set_hit(float t, int code, int primitive, wp::vec3 n) {
  optixSetPayload_0(__float_as_uint(t));
  optixSetPayload_1((unsigned int)code);
  optixSetPayload_2((unsigned int)primitive);
  optixSetPayload_3(__float_as_uint(n[0]));
  optixSetPayload_4(__float_as_uint(n[1]));
  optixSetPayload_5(__float_as_uint(n[2]));
}

// Custom-primitive hit (hit kind 0): world-space normal and world · ngeom + geom as attributes.
__device__ inline bool report_intersection(float t, wp::vec3 n, int code) {
  return optixReportIntersection(t, 0u, __float_as_uint(n[0]), __float_as_uint(n[1]), __float_as_uint(n[2]),
                                 (unsigned int)code);
}

}  // namespace mjpbr_optix
"""


class _TraceResult(ctypes.Structure):
  _fields_ = [
    ("t", ctypes.c_float),
    ("instance", ctypes.c_int32),
    ("primitive", ctypes.c_int32),
    ("a", ctypes.c_float),
    ("b", ctypes.c_float),
    ("c", ctypes.c_float),
  ]


AVAILABLE = hasattr(wp, "build_experimental") and hasattr(wp, "ModuleBuildOptions")

if AVAILABLE:
  _B = wp.build_experimental
  _B.add_native_type(
    _TraceResult,
    native_name="mjpbr_optix::TraceResult",
    fields={"t": float, "instance": wp.int32, "primitive": wp.int32, "a": float, "b": float, "c": float},
  )
  _RAY = {"handle": wp.uint64, "o": wp.vec3, "d": wp.vec3, "tmin": float, "tmax": float}
  _B.add_builtin("mjpbr_optix_launch_index", {}, int, native_name="mjpbr_optix::launch_index")
  _B.add_builtin(
    "mjpbr_optix_trace_closest",
    {**_RAY, "ray_flags": wp.uint32},
    _TraceResult,
    native_name="mjpbr_optix::trace_closest",
  )
  _B.add_builtin("mjpbr_optix_trace_any", _RAY, wp.bool, native_name="mjpbr_optix::trace_any")
  _B.add_builtin("mjpbr_optix_miss", {}, None, native_name="mjpbr_optix::miss")
  _B.add_builtin("mjpbr_optix_closesthit_triangle", {}, None, native_name="mjpbr_optix::closesthit_triangle")
  _B.add_builtin("mjpbr_optix_closesthit_custom", {}, None, native_name="mjpbr_optix::closesthit_custom")
  _B.add_builtin("mjpbr_optix_instance_id", {}, int, native_name="mjpbr_optix::instance_id")
  _B.add_builtin("mjpbr_optix_ray_tmin", {}, float, native_name="mjpbr_optix::ray_tmin")
  _B.add_builtin("mjpbr_optix_ray_tmax", {}, float, native_name="mjpbr_optix::ray_tmax")
  _B.add_builtin("mjpbr_optix_primitive_index", {}, int, native_name="mjpbr_optix::primitive_index")
  _B.add_builtin("mjpbr_optix_world_ray_origin", {}, wp.vec3, native_name="mjpbr_optix::world_ray_origin")
  _B.add_builtin("mjpbr_optix_world_ray_direction", {}, wp.vec3, native_name="mjpbr_optix::world_ray_direction")
  _B.add_builtin(
    "mjpbr_optix_set_hit",
    {"t": float, "code": int, "primitive": int, "n": wp.vec3},
    None,
    native_name="mjpbr_optix::set_hit",
  )
  _B.add_builtin(
    "mjpbr_optix_report_intersection",
    {"t": float, "n": wp.vec3, "code": int},
    wp.bool,
    native_name="mjpbr_optix::report_intersection",
  )


def include_dir() -> Optional[Path]:
  """OptiX SDK include directory (holding optix_device.h), or None if not found.

  Looked up from `OPTIX_INCLUDE_DIR`, then `OPTIX_ROOT` / `OptiX_INSTALL_DIR` (+ /include).
  """
  candidates = []
  if os.environ.get("OPTIX_INCLUDE_DIR"):
    candidates.append(Path(os.environ["OPTIX_INCLUDE_DIR"]))
  for var in ("OPTIX_ROOT", "OptiX_INSTALL_DIR", "OPTIX_INSTALL_DIR"):
    if os.environ.get(var):
      candidates.append(Path(os.environ[var]) / "include")
  for path in candidates:
    if (path / "optix_device.h").exists():
      return path
  return None


def build_options(extra_preamble: str = "") -> "wp.ModuleBuildOptions":
  """Module build options of a module holding OptiX programs."""
  inc = include_dir()
  if inc is None:
    raise RuntimeError("OptiX SDK headers not found: set OPTIX_INCLUDE_DIR to the directory holding optix_device.h")
  return wp.ModuleBuildOptions(
    extra_cuda_include_dirs=[str(inc)],
    extra_cuda_preamble=_PREAMBLE + extra_preamble,
  )
