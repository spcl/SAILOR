import argparse
import numpy as np
from npbench.benchmarks.polybench.k3mm import k3mm as base_module
from npbench.benchmarks.polybench.k3mm import k3mm_numpy as numpy_module
from npbench.benchmarks.polybench.k3mm import k3mm_dace as dace_module

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
        _NM = 1000
    elif preset == 'M':
        _NI = 2000
        _NJ = 2200
        _NK = 2400
        _NL = 2600
        _NM = 2800
    elif preset == 'L':
        _NI = 5500
        _NJ = 6000
        _NK = 6500
        _NL = 7000
        _NM = 7500
    elif preset == 'paper':
        _NI = 3200
        _NJ = 3600
        _NK = 4000
        _NL = 4400
        _NM = 4800

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(NI=_NI, NJ=_NJ, NK=_NK, NL=_NL, NM=_NM)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A, B, C, D = base_module.initialize(_NI, _NJ, _NK, _NL, _NM)
    if validate:
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        C_copy = np.copy(C)
        D_copy = np.copy(D)

        ret_validate = numpy_module.kernel(A_copy, B_copy, C_copy, D_copy)
        ret = sdfg(A=A, B=B, C=C, D=D, **symbols)

        if not np.allclose(ret, ret_validate):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, A=A, B=B, C=C, D=D, **symbols)
