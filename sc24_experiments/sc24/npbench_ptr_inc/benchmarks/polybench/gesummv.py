import argparse
import numpy as np
from npbench.benchmarks.polybench.gesummv import gesummv as base_module
from npbench.benchmarks.polybench.gesummv import gesummv_numpy as numpy_module
from npbench.benchmarks.polybench.gesummv import gesummv_dace as dace_module

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
        _N = 2000
    elif preset == 'M':
        _N = 4000
    elif preset == 'L':
        _N = 14000
    elif preset == 'paper':
        _N = 11200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, A, B, x = base_module.initialize(_N)
    if validate:
        alpha_copy = np.copy(alpha)
        beta_copy = np.copy(beta)
        A_copy = np.copy(A)
        B_copy = np.copy(B)
        x_copy = np.copy(x)

        ret_validate = numpy_module.kernel(alpha_copy, beta_copy, A_copy, B_copy, x_copy)
        ret = sdfg(alpha=alpha, beta=beta, x=x, A=A, B=B, **symbols)

        if not np.allclose(ret_validate, ret):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, alpha=alpha, beta=beta, A=A, B=B, x=x, **symbols)
