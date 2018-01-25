from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


prg = cl.Program(ctx,open('mm2metersKernel.cl').read()).build()



