from time import perf_counter
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
        self._enable_timing = enable_timing
        self._blocking = blocking

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

    # coarse-grained timing
    est = 5
    _t1= perf_counter()
    for _ in range(est):
        fn()
    cuda_device_synchronize()
    _t2 = perf_counter()
    estimate_ms = (_t2-_t1)*1000/est
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # benchmark
    # start_event = [Event(True, True) for i in range(n_repeat)]
    # end_event = [Event(True, True) for i in range(n_repeat)]
    start_event = Event(True, True)
    end_event = Event(True, True)

    # Warm-up
    total = 0
    for _ in range(n_warmup):
        fn()

    # Benchmark
    for _ in range(n_repeat):
        _t1= perf_counter()
        start_event.record(BENCHMARK_STREAM)
        fn()
        end_event.record(BENCHMARK_STREAM)
        _t2 = perf_counter()
        end_event.synchronize()
        total += end_event.elapsed_time(start_event)

    print(f'estimate_ms = {estimate_ms}; number of repeat: {n_repeat}; number of warmup: {n_warmup}')

    return total/n_repeat

    # times = np.array([e.elapsed_time(s) for s, e in zip(start_event, end_event)])
    # if percentiles:
    #     percentiles = np.quantile(times, percentiles)
    #     return tuple(percentiles)
    # else:
    #     return np.mean(times).item()

