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

Errors of the worker cannot propagate through `wp.capture_launch`: they are kept and raised by
`Bridge.raise_error` (the next render, or `mujoco_warp_pbr._src.optix.check`).

Lifetime: a graph holds the addresses of its bridge's semaphores, and may be launched after the
renderer that captured it was replaced or released. The semaphores are therefore never freed, and
a closed bridge's worker keeps answering requests without doing any work (the graph then leaves
the outputs unchanged; `raise_released_error` reports it). Each closed bridge thus keeps one
idle thread.

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
from typing import Protocol

import warp as wp

_SEM_BYTES = 64  # >= sizeof(sem_t)
_CU_STREAM_NON_BLOCKING = 1

_libs = {}
# Semaphores of every bridge, never freed: captured graphs hold their addresses.
_semaphores: list = []
# First replay of a graph whose bridge was closed, not yet reported.
_released_error: list[RuntimeError] = []


class GraphWork(Protocol):
  """Owner of a bridge: the OptiX calls of a frame."""

  def graph_work(self, stream: wp.Stream) -> None: ...


def available() -> bool:
  """Whether the bridge is supported on this platform (Linux; libcuda.so is loaded on first use)."""
  return sys.platform.startswith("linux")


def _load() -> dict:
  if not _libs:
    libc = ctypes.CDLL(None)
    for name in ("sem_init", "sem_post", "sem_wait"):
      getattr(libc, name).argtypes = [ctypes.c_void_p] + ([ctypes.c_int, ctypes.c_uint] if name == "sem_init" else [])
      getattr(libc, name).restype = ctypes.c_int
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cuda.cuLaunchHostFunc.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int
    _libs.update(
      libc=libc,
      cuda=cuda,
      post=ctypes.cast(libc.sem_post, ctypes.c_void_p).value,
      wait=ctypes.cast(libc.sem_wait, ctypes.c_void_p).value,
    )
  return _libs


def _semaphore(libc) -> ctypes.Array:
  sem = ctypes.create_string_buffer(_SEM_BYTES)
  if libc.sem_init(sem, 0, 0):
    raise OSError("sem_init failed")
  _semaphores.append(sem)
  return sem


def raise_released_error() -> None:
  """Raises (once) if a graph was launched after the OptiX renderer that captured it was closed."""
  if _released_error:
    raise _released_error.pop()


class Bridge:
  """Runs `owner.graph_work(stream)` on a worker thread when a graph reaches `enqueue`'s point.

  Holds only a weak reference to its owner; `close` (or the owner's collection) stops the work.
  """

  def __init__(self, owner: GraphWork, device: wp.Device):
    libs = _load()
    self._libc = libs["libc"]
    self._cuda = libs["cuda"]
    self._owner = weakref.ref(owner)
    self._device = device
    # Non-blocking: a blocking stream would also be waited on by legacy-stream work, such as the
    # synchronous copies to pageable host memory behind array.numpy().
    with wp.ScopedDevice(device):
      handle = ctypes.c_void_p()
      result = self._cuda.cuStreamCreate(ctypes.byref(handle), _CU_STREAM_NON_BLOCKING)
      if result != 0:
        raise RuntimeError(f"cuStreamCreate failed (CUresult {result})")
      self._stream_handle = handle.value
      self.stream: wp.Stream | None = wp.Stream(device, cuda_stream=handle.value)
    self._request = _semaphore(self._libc)
    self._done = _semaphore(self._libc)
    # Held while the worker runs the owner's work; reentrant: the owner may be collected (and
    # close its bridge) on the worker thread.
    self._lock = threading.RLock()
    self._closed = False
    self.error: BaseException | None = None
    self._thread = threading.Thread(target=self._run, name="mujoco_warp_pbr-optix-graph", daemon=True)
    self._thread.start()

  def _run(self):
    while True:
      self._libc.sem_wait(self._request)  # ctypes releases the GIL while waiting
      owner = stream = None
      try:
        with self._lock:
          owner = None if self._closed else self._owner()
          stream = self.stream
          if owner is None or stream is None:
            if not _released_error:
              _released_error.append(
                RuntimeError("a CUDA graph was launched after its OptiX renderer was released; its OptiX work was skipped")
              )
            continue
          with wp.ScopedDevice(self._device):
            owner.graph_work(stream)
            wp.synchronize_stream(stream)
      except BaseException as e:  # noqa: BLE001 (reported by raise_error; the graph must not hang)
        self.error = e
      finally:
        # No references while idle: collecting the owner closes this bridge (and its stream).
        stream = None
        owner = None
        self._libc.sem_post(self._done)

  def enqueue(self, stream: wp.Stream) -> None:
    """Records (inside a capture) the host nodes that run the owner's OptiX work."""
    if self._closed:
      raise RuntimeError("the OptiX graph bridge is closed")
    libs = _load()
    handle = ctypes.c_void_p(stream.cuda_stream)
    for fn, sem in ((libs["post"], self._request), (libs["wait"], self._done)):
      result = self._cuda.cuLaunchHostFunc(handle, fn, ctypes.addressof(sem))
      if result != 0:
        raise RuntimeError(f"cuLaunchHostFunc failed (CUresult {result})")

  def raise_error(self) -> None:
    """Re-raises (once) an error of the worker, or of a graph launched after its bridge closed."""
    if self.error is not None:
      error, self.error = self.error, None
      raise RuntimeError("OptiX work of a captured graph failed") from error
    raise_released_error()

  def close(self) -> None:
    """Stops running the owner's work (waiting for a running one) and destroys the stream.

    The worker and the semaphores stay, for graphs captured through this bridge (see the module
    docstring).
    """
    with self._lock:
      if self._closed:
        return
      self._closed = True
      stream, self.stream = self.stream, None
      del stream  # unregisters the handle from Warp
      # Not during a capture (close may run from a collection): the stream is then leaked.
      if not self._device.is_capturing:
        self._cuda.cuStreamDestroy_v2(ctypes.c_void_p(self._stream_handle))

  def __del__(self):
    try:
      self.close()
    except Exception:  # noqa: BLE001 (interpreter shutdown: nothing left to report to)
      pass
