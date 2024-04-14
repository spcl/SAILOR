import argparse
import numpy as np
from npbench.benchmarks.polybench.ludcmp import ludcmp as base_module
from npbench.benchmarks.polybench.ludcmp import ludcmp_numpy as numpy_module
from npbench.benchmarks.polybench.ludcmp import ludcmp_dace as dace_module

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
        _N = 650
    elif preset == 'paper':
        _N = 2000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A, b = base_module.initialize(_N)
    if validate:
        A_copy = np.copy(A)
        b_copy = np.copy(b)

        x_validate, y_validate = numpy_module.kernel(A_copy, b_copy)
        x, y = sdfg(A=A, b=b, **symbols)

        if not np.allclose(x_validate, x) or not np.allclose(y_validate, y):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, b=b, **symbols)
