import argparse
import numpy as np
from npbench.benchmarks.deep_learning.mlp import mlp as base_module
from npbench.benchmarks.deep_learning.mlp import mlp_numpy as numpy_module
from npbench.benchmarks.deep_learning.mlp import mlp_dace as dace_module

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
        _C_in = 3
        _N = 8
        _S0 = 3000
        _S1 = 2000
        _S2 = 2000
    elif preset == 'M':
        _C_in = 3
        _N = 8
        _S0 = 3000
        _S1 = 10000
        _S2 = 10000
    elif preset == 'L':
        _C_in = 3
        _N = 8
        _S0 = 3000
        _S1 = 30000
        _S2 = 30000
    elif preset == 'paper':
        _C_in = 3
        _N = 8
        _S0 = 3000
        _S1 = 10000
        _S2 = 1000

    sdfg = dace_module.mlp.to_sdfg(simplify=False)

    symbols = dict(N=_N, C_in=_C_in, S0=_S0, S1=_S1, S2=_S2)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    (input, w1, b1, w2, b2, w3, b3) = base_module.initialize(_C_in, _N, _S0, _S1, _S2)
    if validate:
        input_copy = np.copy(input)
        w1_copy = np.copy(w1)
        b1_copy = np.copy(b1)
        w2_copy = np.copy(w2)
        b2_copy = np.copy(b2)
        w3_copy = np.copy(w3)
        b3_copy = np.copy(b3)

        ret_validate = numpy_module.mlp(input_copy, w1_copy, b1_copy, w2_copy, b2_copy, w3_copy, b3_copy)
        ret = sdfg(input, w1, b1, w2, b2, w3, b3, **symbols)

        if not np.allclose(ret_validate, ret):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, input=input, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, **symbols)
