import argparse
import numpy as np
from npbench.benchmarks.polybench.seidel_2d import seidel_2d as base_module
from npbench.benchmarks.polybench.seidel_2d import seidel_2d_numpy as numpy_module
from npbench.benchmarks.polybench.seidel_2d import seidel_2d_dace as dace_module

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
        _TSTEPS = 8
        _N = 50
    elif preset == 'M':
        _TSTEPS = 15
        _N = 100
    elif preset == 'L':
        _TSTEPS = 40
        _N = 200
    elif preset == 'paper':
        _TSTEPS = 100
        _N = 400

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, TSTEPS=_TSTEPS)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A = base_module.initialize(_N)
    if validate:
        A_copy = np.copy(A)

        numpy_module.kernel(_TSTEPS, _N, A_copy)
        sdfg(A=A, **symbols)

        if not np.allclose(A, A_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, **symbols)
