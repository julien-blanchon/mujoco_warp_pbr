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

"""CUDA graph capture of the OptiX calls through a host-function bridge.

`optixLaunch` cannot be stream-captured (OptiX 9.0 on RTX, 9.1 on GH200). Traced with Nsight
Systems, an eager launch is one `cuMemcpyDtoDAsync` (the launch parameters) plus three
`cuEventRecord`s, and the ray-tracing kernel is submitted by the driver outside the CUDA API, under
the memcpy's correlation id. Inside a capture the memcpy and the event records are captured, but the
kernel still runs immediately, at capture time, reading a parameter slot that the captured copy
never filled; the graph holds only the memcpy node. Stream capture mode (global, thread-local,
relaxed) makes no difference. Acceleration-structure builds are captured, but OptiX then fails on
its next eager build ("Failed to query command list event"), so they cannot be captured either.

Inside a capture the renderer therefore records two host nodes in place of its OptiX calls:
`sem_post` (libc) wakes a worker thread, and `sem_wait` blocks the graph until the worker has
run the OptiX calls (acceleration-structure refits and `optixLaunch`) on its own stream and
synchronised it. Everything before the host nodes (instance and parameter writes) and after them
(supersampling) stays in the graph. The host nodes call only libc semaphore functions, never CUDA
or Python; the worker needs the GIL for its few PyOptiX calls, which it gets whenever the host
thread waits on the GPU (Warp, PyTorch and NumPy synchronisation release the GIL).

Caveat (hence opt-in, `PbrRenderContext.optix_graph_capture`): a synchronous copy to pageable host
memory (`array.numpy()`) issued while such a graph runs blocks inside the driver holding a lock
that the worker's kernel launches need, while the copy itself waits for the graph: a deadlock.
Synchronising first (`wp.synchronize()`, which waits without that lock) avoids it. The same
deadlock rules out every variant in which the GPU waits for a later host-side OptiX call. Gating
pre-enqueued OptiX work on the GPU instead (`cuStreamWaitValue32`, no host on the critical path)
does not deadlock that copy but makes `wp.synchronize()` wait forever for the pending work.

Linux only (libc semaphores, libcuda.so).
"""

import ctypes
import sys
import threading
import weakref

import warp as wp

_libs = {}


def available() -> bool:
  """Whether the bridge can run here (Linux: libc semaphores and libcuda.so)."""
  return sys.platform.startswith("linux")


def _load():
  if not _libs:
    libc = ctypes.CDLL(None)
    for name in ("sem_init", "sem_post", "sem_wait", "sem_destroy"):
      getattr(libc, name).argtypes = [ctypes.c_void_p] + ([ctypes.c_int, ctypes.c_uint] if name == "sem_init" else [])
      getattr(libc, name).restype = ctypes.c_int
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cuda.cuLaunchHostFunc.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    _libs.update(
      libc=libc,
      cuda=cuda,
      post=ctypes.cast(libc.sem_post, ctypes.c_void_p).value,
      wait=ctypes.cast(libc.sem_wait, ctypes.c_void_p).value,
    )
  return _libs


class Bridge:
  """Runs `owner.graph_work(stream)` on a worker thread when a graph reaches `enqueue`'s point.

  Holds only a weak reference to its owner; `close` (or the owner's collection) stops the worker.
  """

  _SEM_BYTES = 64  # >= sizeof(sem_t)

  def __init__(self, owner, device: "wp.Device"):
    libs = _load()
    self._libc = libs["libc"]
    self._owner = weakref.ref(owner)
    self._device = device
    # Non-blocking: a blocking stream would also be waited on by legacy-stream work, such as the
    # synchronous copies to pageable host memory behind array.numpy().
    with wp.ScopedDevice(device):
      handle = ctypes.c_void_p()
      result = libs["cuda"].cuStreamCreate(ctypes.byref(handle), 1)  # CU_STREAM_NON_BLOCKING
      if result != 0:
        raise RuntimeError(f"cuStreamCreate failed (CUresult {result})")
      self.stream = wp.Stream(device, cuda_stream=handle.value)
    self._request = ctypes.create_string_buffer(self._SEM_BYTES)
    self._done = ctypes.create_string_buffer(self._SEM_BYTES)
    if self._libc.sem_init(self._request, 0, 0) or self._libc.sem_init(self._done, 0, 0):
      raise OSError("sem_init failed")
    self._closed = False
    self.error = None
    self._thread = threading.Thread(target=self._run, name="mujoco_warp_pbr-optix-graph", daemon=True)
    self._thread.start()

  def _run(self):
    while True:
      self._libc.sem_wait(self._request)  # ctypes releases the GIL while waiting
      if self._closed:
        self._libc.sem_post(self._done)
        return
      try:
        owner = self._owner()
        if owner is None:
          raise RuntimeError("the OptiX renderer of a captured graph was released")
        with wp.ScopedDevice(self._device):
          owner.graph_work(self.stream)
          wp.synchronize_stream(self.stream)
      except BaseException as e:  # reported by the next render; the graph must not hang
        self.error = e
      finally:
        self._libc.sem_post(self._done)

  def enqueue(self, stream: "wp.Stream"):
    """Records (inside a capture) the host nodes that run the owner's OptiX work."""
    libs = _load()
    handle = ctypes.c_void_p(stream.cuda_stream)
    for fn, sem in ((libs["post"], self._request), (libs["wait"], self._done)):
      result = libs["cuda"].cuLaunchHostFunc(handle, fn, ctypes.addressof(sem))
      if result != 0:
        raise RuntimeError(f"cuLaunchHostFunc failed (CUresult {result})")

  def raise_error(self):
    """Re-raises (once) an error of the worker."""
    if self.error is not None:
      error, self.error = self.error, None
      raise RuntimeError("OptiX work of a captured graph failed") from error

  def close(self):
    if not self._closed:
      self._closed = True
      self._libc.sem_post(self._request)
      self._thread.join(timeout=5.0)

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
