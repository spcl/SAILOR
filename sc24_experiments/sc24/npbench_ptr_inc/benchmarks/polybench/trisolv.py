import argparse
import numpy as np
from npbench.benchmarks.polybench.trisolv import trisolv as base_module
from npbench.benchmarks.polybench.trisolv import trisolv_numpy as numpy_module
from npbench.benchmarks.polybench.trisolv import trisolv_dace as dace_module

from sc24.npbench_ptr_inc import utils

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-r' '--repetitions', dest='repetitions', type=int, default=1000)
    argp.add_argument('-p' '--preset', dest='preset', type=str, default='S')
    argp.add_argument('--increment', dest='increment', default=False, action='store_true')
    argp.add_argument('--parallel', dest='parallel', default=False, action='store_true')
    argp.add_argument('--validate', dest='validate', default=False, action='store_true')
    args = argp.parse_args()

    T_REPS = args.repetitions

    increment = args.increment
    parallel = args.parallel
    preset = args.preset
    validate = args.validate

    if preset == 'S':
        _N = 2000
    elif preset == 'M':
        _N = 5000
    elif preset == 'L':
        _N = 14000
    elif preset == 'paper':
        _N = 16000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    L, x, b = base_module.initialize(_N)
    if validate:
        L_copy = np.copy(L)
        x_copy = np.copy(x)
        b_copy = np.copy(b)

        numpy_module.kernel(L_copy, x_copy, b_copy)
        sdfg(L=L, x=x, b=b, **symbols)

        if not np.allclose(x, x_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, L=L, x=x, b=b, **symbols)
