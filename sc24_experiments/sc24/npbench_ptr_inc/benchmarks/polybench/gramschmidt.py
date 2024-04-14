import argparse
import numpy as np
from npbench.benchmarks.polybench.gramschmidt import gramschmidt as base_module
from npbench.benchmarks.polybench.gramschmidt import gramschmidt_numpy as numpy_module
from npbench.benchmarks.polybench.gramschmidt import gramschmidt_dace as dace_module

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
    validate= args.validate

    if preset == 'S':
        _M = 70
        _N = 60
    elif preset == 'M':
        _M = 220
        _N = 180
    elif preset == 'L':
        _M = 600
        _N = 500
    elif preset == 'paper':
        _M = 240
        _N = 200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(M=_M, N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A = base_module.initialize(_M, _N)
    if validate:
        A_copy = np.copy(A)

        Q_validate, R_validate = numpy_module.kernel(A_copy)
        Q, R = sdfg(A=A, **symbols)

        if not np.allclose(Q, Q_validate) or not np.allclose(R, R_validate):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, **symbols)
