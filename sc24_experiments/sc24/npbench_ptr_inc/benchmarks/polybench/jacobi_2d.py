import argparse
import numpy as np
from npbench.benchmarks.polybench.jacobi_2d import jacobi_2d as base_module
from npbench.benchmarks.polybench.jacobi_2d import jacobi_2d_numpy as numpy_module
from npbench.benchmarks.polybench.jacobi_2d import jacobi_2d_dace as dace_module

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
        _TSTEPS = 50
        _N = 150
    elif preset == 'M':
        _TSTEPS = 80
        _N = 350
    elif preset == 'L':
        _TSTEPS = 200
        _N = 700
    elif preset == 'paper':
        _TSTEPS = 1000
        _N = 2800

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, TSTEPS=_TSTEPS)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A, B = base_module.initialize(_N)
    if validate:
        A_copy = np.copy(A)
        B_copy = np.copy(B)

        numpy_module.kernel(_TSTEPS, A_copy, B_copy)
        sdfg(TSTEPS=_TSTEPS, A=A, B=B, N=_N)

        if not np.allclose(A, A_copy) or not np.allclose(B, B_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, TSTEPS=_TSTEPS, A=A, B=B, N=_N)
