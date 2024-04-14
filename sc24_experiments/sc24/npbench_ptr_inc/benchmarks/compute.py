import argparse
import numpy as np
from npbench.benchmarks.compute import compute as base_module
from npbench.benchmarks.compute import compute_numpy as numpy_module
from npbench.benchmarks.compute import compute_dace as dace_module

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
        _M = 2000
        _N = 2000
    elif preset == 'M':
        _M = 5000
        _N = 5000
    elif preset == 'L':
        _M = 16000
        _N = 16000
    elif preset == 'paper':
        _M = 12500
        _N = 12500

    sdfg = dace_module.compute.to_sdfg(simplify=False)

    symbols = dict(M=_M, N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    array_1, array_2, a, b, c = base_module.initialize(_M, _N)
    if validate:
        array_1_copy = np.copy(array_1)
        array_2_copy = np.copy(array_2)
        c_copy = np.copy(c)
        b_copy = np.copy(b)
        a_copy = np.copy(a)

        ret_validate = numpy_module.compute(array_1_copy, array_2_copy, a_copy, b_copy, c_copy)
        ret= sdfg(array_1=array_1, array_2=array_2, a=a, b=b, c=c, **symbols)

        if not np.allclose(ret, ret_validate):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, array_1=array_1, array_2=array_2, a=a, b=b, c=c, **symbols)
