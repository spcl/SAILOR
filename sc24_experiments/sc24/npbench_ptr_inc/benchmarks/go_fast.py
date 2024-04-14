import argparse
import numpy as np
from npbench.benchmarks.go_fast import go_fast as base_module
from npbench.benchmarks.go_fast import go_fast_numpy as numpy_module
from npbench.benchmarks.go_fast import go_fast_dace as dace_module

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
        _N = 2000
    elif preset == 'M':
        _N = 6000
    elif preset == 'L':
        _N = 20000
    elif preset == 'paper':
        _N = 12500

    sdfg = dace_module.go_fast.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    x = base_module.initialize(_N)
    if validate:
        x_copy = np.copy(x)

        ret_validate = numpy_module.go_fast(x_copy)
        ret = sdfg(x, **symbols)

        if not np.allclose(ret, ret_validate):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, a=x, **symbols)
