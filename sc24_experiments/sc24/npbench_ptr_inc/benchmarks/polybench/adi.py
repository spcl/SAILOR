import argparse
import numpy as np
from npbench.benchmarks.polybench.adi import adi as base_module
from npbench.benchmarks.polybench.adi import adi_numpy as numpy_module
from npbench.benchmarks.polybench.adi import adi_dace as dace_module

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
        _TSTEPS = 5
        _N = 100
    elif preset == 'M':
        _TSTEPS = 20
        _N = 200
    elif preset == 'L':
        _TSTEPS = 50
        _N = 500
    elif preset == 'paper':
        _TSTEPS = 100
        _N = 200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, TSTEPS=_TSTEPS)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    u = base_module.initialize(_N)
    if validate:
        u_copy = np.copy(u)
        numpy_module.kernel(_TSTEPS, _N, u_copy)
        sdfg(u=u, **symbols)

        if not np.allclose(u, u_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, u=u, **symbols)
