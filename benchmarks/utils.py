import numpy as np

from cuda import cudart
from cuda.cudart import cudaEvent_t, cudaStream_t

def cuda_device_synchronize():
    (err,) = cudart.cudaDeviceSynchronize()
    if err != 0:
        raise RuntimeError("cudaDeviceSynchronize failed with error: {}".format(err.name))


class Stream:
    def __init__(self, blocking: bool = False, priority: int = 0, **kwargs):
        flags = cudart.cudaStreamNonBlocking if not blocking else cudart.cudaStreamDefault
        err, handle = cudart.cudaStreamCreateWithPriority(flags, priority)
        assert err == 0, err
        self._handle = handle

    def __int__(self):
        return int(self._handle)

    def __hash__(self):
        return hash(self._handle)

    def __eq__(self, other):
        raise TypeError(f"cannot compare Stream with {type(other)}")

    def __del__(self):
        # XXX: rely on system to automatically cleanup
        pass
        # (err,) = cudart.cudaStreamDestroy(self._handle)
        # assert err == 0, err

    def handle(self) -> cudaStream_t:
        return self._handle


class Event:
    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self._enable_timing: bool = enable_timing
        self._blocking: bool = blocking

        self._handle: cudaEvent_t
        if not enable_timing:
            flags = cudart.cudaEventDisableTiming
        else:
            flags = cudart.cudaEventDefault
        err, self._handle = cudart.cudaEventCreateWithFlags(flags)
        assert err == 0, err

    def __del__(self):
        (err,) = cudart.cudaEventDestroy(self._handle)
        assert err == 0, err

    def handle(self) -> cudaEvent_t:
        return self._handle

    def elapsed_time(self, start_event) -> float:
        if not self._enable_timing or not start_event._enable_timing:
            raise RuntimeError("Event does not have timing enabled")
        err, elapsed_time = cudart.cudaEventElapsedTime(start_event._handle, self._handle)
        assert err == 0, err
        return elapsed_time

    def record(self, stream):
        if not isinstance(stream, Stream):
            raise TypeError("stream must be a Stream")
        (err,) = cudart.cudaEventRecord(self._handle, stream.handle())
        assert err == 0, err

    def synchronize(self):
        if not self._blocking:
            raise RuntimeError("Event does not have blocking enabled")
        (err,) = cudart.cudaEventSynchronize(self._handle)
        if err != 0:
            raise RuntimeError("cudaEventSynchronize failed with error: {}".format(err.name))

BENCHMARK_STREAM = Stream()

def benchmark(fn, warmup=25, rep=100, percentiles=(0.2, 0.5, 0.8)):

    fn()
    cuda_device_synchronize()

    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    start_event.record(BENCHMARK_STREAM)
    for _ in range(5):
        fn()
    end_event.record(BENCHMARK_STREAM)
    cuda_device_synchronize()

    estimate_ms = end_event.elapsed_time(start_event) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    start_event = [Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        start_event[i].record(BENCHMARK_STREAM)
        fn()
        end_event[i].record(BENCHMARK_STREAM)

    # Record clocks
    cuda_device_synchronize()

    print(f'number of repeat: {n_repeat}; number of warmup: {n_warmup}')
    times = np.array([e.elapsed_time(s) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = np.quantile(times, percentiles)
        return tuple(percentiles)
    else:
        return np.mean(times).item()

