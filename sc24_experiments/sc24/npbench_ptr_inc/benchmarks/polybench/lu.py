import argparse
import numpy as np
from npbench.benchmarks.polybench.lu import lu as base_module
from npbench.benchmarks.polybench.lu import lu_numpy as numpy_module
from npbench.benchmarks.polybench.lu import lu_dace as dace_module

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
        _N = 60
    elif preset == 'M':
        _N = 220
    elif preset == 'L':
        _N = 700
    elif preset == 'paper':
        _N = 2000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A = base_module.initialize(_N)
    if validate:
        A_copy = np.copy(A)

        numpy_module.kernel(A_copy)
        sdfg(A=A, **symbols)

        if not np.allclose(A_copy, A):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, **symbols)
