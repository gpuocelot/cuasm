#!/usr/bin/env python3
# coding: utf-8

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np


import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pycuda.tools
import pycuda.autoinit

from utils import benchmark

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("fn", "out", "cubin file name")
flags.DEFINE_string("func", "_Z9vectorAddPfS_S_i", "func name after mingling")
flags.DEFINE_integer("thr", 5, "thread num")
flags.DEFINE_integer("bx", 256, "block dim x")

def main(_):

    mod = drv.module_from_file(f'{FLAGS.fn}.cubin')
    kernel = mod.get_function(f'{FLAGS.func}')

    for _ in range(10):  # run 10 for verification
        # args short cut (without short cut we can do cu_memcpy explicitly)
        # TODO find a way to get args and cuda launch args from cubin automatically??
        n_element = 1_000_000
        a = np.random.randn(n_element).astype(np.float32)
        b = np.random.randn(n_element).astype(np.float32)
        dest = np.zeros_like(a)
        n = np.int32(n_element)

        # kernel configs
        block = (FLAGS.bx, 1, 1)
        grid = (int(np.ceil((n_element+FLAGS.bx)//FLAGS.bx)), 1)
        print(f'Grid: {grid}; Block: {block}')

        # launch
        kernel(drv.In(a), drv.In(b), drv.Out(dest),  n, #
                block=block, grid=grid, #
        )

        # verify result
        # print('CUBIN:')
        c_ref = a+b
        np.testing.assert_allclose(dest, c_ref)  # if not pass, will raise
        # print(f'{FLAGS.fn} passed')


    # benchmark
    a_gpu = drv.mem_alloc(a.nbytes)
    b_gpu = drv.mem_alloc(b.nbytes)
    dest_gpu = drv.mem_alloc(dest.nbytes)
    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(dest_gpu, dest)

    result = benchmark(fn=lambda: kernel(a_gpu, b_gpu, dest_gpu,  n, #
            block=block, grid=grid, #
    ), #
    warmup=25, rep=100, #
    # percentiles=(0.2, 0.5, 0.8),
    percentiles=None,
    )
    print('BENCHMARK:')
    print(result)


if __name__ == "__main__":
    app.run(main)
