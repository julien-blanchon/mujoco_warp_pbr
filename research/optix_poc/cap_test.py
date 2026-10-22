"""Can optixAccelBuild (BUILD_OPERATION_UPDATE) be captured into a CUDA graph? Try several variants."""

import numpy as np, optix, warp as wp, wpx, sys
from warp._src.context import CaptureMode

ctx = wpx.create_context()
vs, fs = wpx.load_obj(wpx.HELMET)
verts = wp.array(vs, dtype=wp.vec3)
inds3 = wp.array(fs, dtype=wp.int32)
gas = wpx.build_accel(ctx, [wpx.triangle_input(verts, inds3)], optix.BUILD_FLAG_PREFER_FAST_TRACE)
inst = np.zeros(8, dtype=wpx.OptixInstance.numpy_dtype())
for k in range(8):
  inst[k]["t0"] = (1, 0, 0, k)
  inst[k]["t1"] = (0, 1, 0, 0)
  inst[k]["t2"] = (0, 0, 1, 0)
  inst[k]["visibility_mask"] = 255
  inst[k]["handle"] = gas[0]
ib = wp.array(inst, dtype=wpx.OptixInstance)
fl = optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE
bi = [optix.BuildInputInstanceArray(instances=ib.ptr, numInstances=8)]
variant = sys.argv[1]
st = wp.Stream() if "newstream" in variant else wp.get_stream()
with wp.ScopedStream(st):
  a = wpx.build_accel(ctx, bi, fl)
  ou = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_UPDATE)
  for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 0):
    ctx.accelBuild(st.cuda_stream, [ou], bi, a[2].ptr, a[2].capacity, a[1].ptr, a[1].capacity, [])
  wp.synchronize()
  ou = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_UPDATE)
  mode = (
    CaptureMode.RELAXED if "relaxed" in variant else CaptureMode.GLOBAL if "global" in variant else CaptureMode.THREAD_LOCAL
  )
  try:
    with wp.ScopedCapture(stream=st, capture_mode=mode) as cap:
      for _ in range(int(sys.argv[2])):
        ctx.accelBuild(st.cuda_stream, [ou], bi, a[2].ptr, a[2].capacity, a[1].ptr, a[1].capacity, [])
    wp.capture_launch(cap.graph)
    wp.synchronize()
    print(variant, sys.argv[2], "CAPTURE OK")
  except Exception as e:
    print(variant, sys.argv[2], "CAPTURE FAILED", repr(e))
