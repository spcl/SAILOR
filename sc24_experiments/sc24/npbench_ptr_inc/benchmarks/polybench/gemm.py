import argparse
import numpy as np
from npbench.benchmarks.polybench.gemm import gemm
from npbench.benchmarks.polybench.gemm import gemm_numpy as numpy_module
from npbench.benchmarks.polybench.gemm import gemm_dace

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
        _NI = 1000
        _NJ = 1100
        _NK = 1200
    elif preset == 'M':
        _NI = 2500
        _NJ = 2750
        _NK = 3000
    elif preset == 'L':
        _NI = 7000
        _NJ = 7500
        _NK = 8000
    elif preset == 'paper':
        _NI = 2000
        _NJ = 2300
        _NK = 2600

    sdfg = gemm_dace.kernel.to_sdfg(simplify=False)

    symbols = dict(NI=_NI, NJ=_NJ, NK=_NK)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, C, A, B = gemm.initialize(_NI, _NJ, _NK)
    if validate:
        alpha_copy = np.copy(alpha)
        beta_copy = np.copy(beta)
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        C_copy = np.copy(C)

        numpy_module.kernel(alpha_copy, beta_copy, C_copy, A_copy, B_copy)
        sdfg(alpha=alpha, beta=beta, C=C, A=A, B=B, **symbols)

        if not np.allclose(C_copy, C):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, alpha=alpha, beta=beta, C=C, A=A, B=B, **symbols)
