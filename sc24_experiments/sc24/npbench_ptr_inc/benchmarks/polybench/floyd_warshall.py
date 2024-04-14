import argparse
import numpy as np
from npbench.benchmarks.polybench.floyd_warshall import floyd_warshall as base_module
from npbench.benchmarks.polybench.floyd_warshall import floyd_warshall_numpy as numpy_module
from npbench.benchmarks.polybench.floyd_warshall import floyd_warshall_dace as dace_module

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
        _N = 200
    elif preset == 'M':
        _N = 400
    elif preset == 'L':
        _N = 850
    elif preset == 'paper':
        _N = 2800

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    path = base_module.initialize(_N)
    if validate:
        path_copy = np.copy(path)

        numpy_module.kernel(path_copy)
        sdfg(path=path, **symbols)

        if not np.allclose(path_copy, path):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, path=path, **symbols)
