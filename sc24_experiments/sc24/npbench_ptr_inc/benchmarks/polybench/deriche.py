import argparse
import numpy as np
from npbench.benchmarks.polybench.deriche import deriche as base_module
from npbench.benchmarks.polybench.deriche import deriche_numpy as numpy_module
from npbench.benchmarks.polybench.deriche import deriche_dace as dace_module

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
        _W = 400
        _H = 200
    elif preset == 'M':
        _W = 1500
        _H = 1000
    elif preset == 'L':
        _W = 6000
        _H = 3000
    elif preset == 'paper':
        _W = 7680
        _H = 4320

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(W=_W, H=_H)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, imgIn = base_module.initialize(_W, _H)
    if validate:
        alpha_copy = np.copy(alpha)
        imgIn_copy = np.copy(imgIn)
        imgOut_validate = numpy_module.kernel(alpha_copy, imgIn_copy)
        imgOut = sdfg(alpha=alpha, imgIn=imgIn, **symbols)

        if not np.allclose(imgOut_validate, imgOut):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, alpha=alpha, imgIn=imgIn, **symbols)
