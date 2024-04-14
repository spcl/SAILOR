import argparse
import numpy as np
from npbench.benchmarks.polybench.fdtd_2d import fdtd_2d as base_module
from npbench.benchmarks.polybench.fdtd_2d import fdtd_2d_numpy as numpy_module
from npbench.benchmarks.polybench.fdtd_2d import fdtd_2d_dace as dace_module

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
        _TMAX = 20
        _NX = 200
        _NY = 220
    elif preset == 'M':
        _TMAX = 60
        _NX = 400
        _NY = 450
    elif preset == 'L':
        _TMAX = 150
        _NX = 800
        _NY = 900
    elif preset == 'paper':
        _TMAX = 500
        _NX = 1000
        _NY = 1200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(TMAX=_TMAX, NX=_NX, NY=_NY)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    ex, ey, hz, _fict_ = base_module.initialize(_TMAX, _NX, _NY)
    if validate:
        ex_copy = np.copy(ex)
        ey_copy = np.copy(ey)
        hz_copy = np.copy(hz)
        _fict__copy = np.copy(_fict_)

        numpy_module.kernel(_TMAX, ex_copy, ey_copy, hz_copy, _fict__copy)
        sdfg(ex=ex, ey=ey, hz=hz, _fict_=_fict_, **symbols)

        if not np.allclose(ey_copy, ey) or not np.allclose(ex_copy, ex) or not np.allclose(hz_copy, hz) or not np.allclose(_fict__copy, _fict_):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, ex=ex, ey=ey, hz=hz, _fict_=_fict_, **symbols)
