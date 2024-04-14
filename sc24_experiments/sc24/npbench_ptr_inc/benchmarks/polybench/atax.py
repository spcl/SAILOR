import argparse
import numpy as np
from npbench.benchmarks.polybench.atax import atax as base_module
from npbench.benchmarks.polybench.atax import atax_numpy as numpy_module
from npbench.benchmarks.polybench.atax import atax_dace as dace_module

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
        _M = 4000
        _N = 5000
    elif preset == 'M':
        _M = 10000
        _N = 12500
    elif preset == 'L':
        _M = 20000
        _N = 25000
    elif preset == 'paper':
        _M = 18000
        _N = 22000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, M=_M)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    x, A = base_module.initialize(_M, _N)
    if validate:
        x_copy = np.copy(x)
        A_copy = np.copy(A)
        ret_validate = numpy_module.kernel(A, x)
        ret = sdfg(x=x, A=A, **symbols)

        if not np.allclose(ret_validate, ret):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, x=x, A=A, **symbols)
