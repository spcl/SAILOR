import argparse
import numpy as np
from npbench.benchmarks.polybench.mvt import mvt as base_module
from npbench.benchmarks.polybench.mvt import mvt_numpy as numpy_module
from npbench.benchmarks.polybench.mvt import mvt_dace as dace_module

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
        _N = 5500
    elif preset == 'M':
        _N = 11000
    elif preset == 'L':
        _N = 22000
    elif preset == 'paper':
        _N = 16000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    x1, x2, y_1, y_2, A = base_module.initialize(_N)
    if validate:
        A_copy = np.copy(A)
        x1_copy = np.copy(x1)
        x2_copy = np.copy(x2)
        y_1_copy = np.copy(y_1)
        y_2_copy = np.copy(y_2)

        numpy_module.kernel(x1_copy, x2_copy, y_1_copy, y_2_copy, A_copy)
        sdfg(x1=x1, x2=x2, y_1=y_1, y_2=y_2, A=A, **symbols)

        if not np.allclose(x1_copy, x1) or not np.allclose(x2_copy, x2):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, x1=x1, x2=x2, y_1=y_1, y_2=y_2, A=A, **symbols)
