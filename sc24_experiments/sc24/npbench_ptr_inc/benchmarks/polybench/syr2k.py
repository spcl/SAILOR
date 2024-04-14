import argparse
import numpy as np
from npbench.benchmarks.polybench.syr2k import syr2k as base_module
from npbench.benchmarks.polybench.syr2k import syr2k_numpy as numpy_module
from npbench.benchmarks.polybench.syr2k import syr2k_dace as dace_module
from sympy import N

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
        _M = 35
        _N = 50
    elif preset == 'M':
        _M = 110
        _N = 140
    elif preset == 'L':
        _M = 350
        _N = 400
    elif preset == 'paper':
        _M = 1000
        _N = 1200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, M=_M)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, C, A, B = base_module.initialize(_M, _N)
    if validate:
        alpha_copy = np.copy(alpha)
        beta_copy = np.copy(beta)
        C_copy = np.copy(C)
        A_copy = np.copy(A)
        B_copy = np.copy(B)

        numpy_module.kernel(alpha_copy, beta_copy, C_copy, A_copy, B_copy)
        sdfg(A=A, C=C, B=B, alpha=alpha, beta=beta, **symbols)

        if not np.allclose(C, C_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, C=C, B=B, alpha=alpha, beta=beta, **symbols)
