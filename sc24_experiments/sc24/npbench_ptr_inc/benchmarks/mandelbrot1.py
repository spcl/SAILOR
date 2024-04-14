import argparse
import numpy as np
from npbench.benchmarks.mandelbrot1 import mandelbrot1_dace as dace_module
from npbench.benchmarks.mandelbrot1 import mandelbrot1_numpy as numpy_module

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
        _xmin = -1.75
        _xmax = 0.25
        _XN = 125
        _ymin = -1.00
        _ymax = 1.00
        _YN = 125
        _maxiter = 60
        _horizon = 2.0
    elif preset == 'M':
        _xmin = -1.75
        _xmax = 0.25
        _XN = 250
        _ymin = -1.00
        _ymax = 1.00
        _YN = 250
        _maxiter = 150
        _horizon = 2.0
    elif preset == 'L':
        _xmin = -2.00
        _xmax = 0.50
        _XN = 833
        _ymin = -1.25
        _ymax = 1.25
        _YN = 833
        _maxiter = 200
        _horizon = 2.0
    elif preset == 'paper':
        _xmin = -2.25
        _xmax = 0.75
        _XN = 1000
        _ymin = -1.25
        _ymax = 1.25
        _YN = 1000
        _maxiter = 200
        _horizon = 2.0

    sdfg = dace_module.mandelbrot.to_sdfg(simplify=False)

    symbols = dict(xmin=_xmin, xmax=_xmax, XN=_XN, ymin=_ymin, ymax=_ymax,
                   YN=_YN, maxiter=_maxiter, horizon=_horizon)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    if validate:
        Z_validate, N_validate = numpy_module.mandelbrot(xn=_XN, yn=_YN, xmin=_xmin,
                                                         xmax=_xmax, ymin=_ymin,
                                                         ymax=_ymax, maxiter=_maxiter, horizon=_horizon)
        Z, N = sdfg(**symbols)

        if not np.allclose(Z_validate, Z) or not np.allclose(N_validate, N):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, **symbols)
