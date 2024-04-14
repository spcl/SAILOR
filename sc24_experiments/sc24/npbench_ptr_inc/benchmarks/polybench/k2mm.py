import argparse
import numpy as np
from npbench.benchmarks.polybench.k2mm import k2mm as base_module
from npbench.benchmarks.polybench.k2mm import k2mm_numpy as numpy_module
from npbench.benchmarks.polybench.k2mm import k2mm_dace as dace_module

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
        _NI = 800
        _NJ = 850
        _NK = 900
        _NL = 950
    elif preset == 'M':
        _NI = 2000
        _NJ = 2250
        _NK = 2500
        _NL = 2750
    elif preset == 'L':
        _NI = 6000
        _NJ = 6500
        _NK = 7000
        _NL = 7500
    elif preset == 'paper':
        _NI = 3200
        _NJ = 3600
        _NK = 4400
        _NL = 4800

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(NI=_NI, NJ=_NJ, NK=_NK, NL=_NL)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, A, B, C, D = base_module.initialize(_NI, _NJ, _NK, _NL)
    if validate:
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        C_copy = np.copy(C)
        D_copy = np.copy(D)
        alpha_copy = np.copy(alpha)
        beta_copy = np.copy(beta)

        numpy_module.kernel(alpha_copy, beta_copy, A_copy, B_copy, C_copy, D_copy)
        sdfg(alpha=alpha, beta=beta, A=A, B=B, C=C, D=D, **symbols)

        if not np.allclose(D, D_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, alpha=alpha, beta=beta, A=A, B=B, C=C, D=D, **symbols)
