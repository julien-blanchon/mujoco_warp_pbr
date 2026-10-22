// Minimal Warp <-> OptiX device glue. Included through ModuleBuildOptions.extra_cuda_preamble,
// i.e. after Warp's own native headers, so wp::vec3 etc. are visible.
#pragma once
// Warp's NVRTC compile has no CUDA toolkit include path: include only the device API (optix.h would pull
// optix_host.h -> <cuda.h>) and skip cooperative vectors (they include <cuda_fp16.h>).
#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 0
#include <optix_device.h>

namespace wpx {

__device__ inline wp::vec3i launch_index()
{
    const uint3 i = optixGetLaunchIndex();
    return wp::vec3i(int(i.x), int(i.y), int(i.z));
}

__device__ inline wp::vec3i launch_dims()
{
    const uint3 d = optixGetLaunchDimensions();
    return wp::vec3i(int(d.x), int(d.y), int(d.z));
}

// optixTrace with two 32-bit payload registers: p0 = hit distance (float bits), p1 = prim/instance code.
// Returns t (or -1 on miss as set by the miss program); prim id written through an out-param style
// helper is awkward in Warp, so the second payload is fetched with a separate call (see trace2).
__device__ inline wp::vec2 trace_t_prim(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                        wp::uint32 ray_flags, wp::uint32 vis_mask)
{
    unsigned int p0 = 0u, p1 = 0u;
    optixTrace(OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]),
               tmin, tmax, 0.0f, OptixVisibilityMask(vis_mask), ray_flags,
               0u /*SBT offset*/, 1u /*SBT stride*/, 0u /*miss index*/, p0, p1);
    return wp::vec2(__uint_as_float(p0), float(p1));
}

// Any-hit style occlusion query (terminate on first hit, no closest-hit program). p0 = 1 if occluded.
__device__ inline bool trace_occluded(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax)
{
    unsigned int p0 = 1u;  // miss program sets 0
    optixTrace(OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]),
               tmin, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0u, 1u, 0u, p0);
    return p0 != 0u;
}

// Program-side accessors for hit / miss programs.
__device__ inline void set_payload_0(wp::uint32 v) { optixSetPayload_0(v); }
__device__ inline void set_payload_1(wp::uint32 v) { optixSetPayload_1(v); }
__device__ inline wp::uint32 get_payload_0() { return optixGetPayload_0(); }
__device__ inline float ray_tmax() { return optixGetRayTmax(); }
__device__ inline wp::uint32 primitive_index() { return optixGetPrimitiveIndex(); }
__device__ inline wp::uint32 instance_id() { return optixGetInstanceId(); }
__device__ inline wp::vec2 triangle_barycentrics()
{
    const float2 b = optixGetTriangleBarycentrics();
    return wp::vec2(b.x, b.y);
}
__device__ inline wp::vec3 object_ray_origin()
{
    const float3 v = optixGetObjectRayOrigin();
    return wp::vec3(v.x, v.y, v.z);
}
__device__ inline wp::vec3 object_ray_direction()
{
    const float3 v = optixGetObjectRayDirection();
    return wp::vec3(v.x, v.y, v.z);
}
__device__ inline float ray_tmin() { return optixGetRayTmin(); }
// Custom-primitive intersection reporting (hit kind 0, no attributes).
__device__ inline bool report_intersection(float t) { return optixReportIntersection(t, 0u); }
__device__ inline void ignore_intersection() { optixIgnoreIntersection(); }
__device__ inline void terminate_ray() { optixTerminateRay(); }
__device__ inline wp::uint32 float_bits(float f) { return __float_as_uint(f); }

}  // namespace wpx
