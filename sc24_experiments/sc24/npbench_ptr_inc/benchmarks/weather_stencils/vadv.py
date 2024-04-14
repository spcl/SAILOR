import argparse
import numpy as np
from npbench.benchmarks.weather_stencils.vadv import vadv as base_module
from npbench.benchmarks.weather_stencils.vadv import vadv_numpy as numpy_module
from npbench.benchmarks.weather_stencils.vadv import vadv_dace as dace_module

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
        _I = 60
        _J = 60
        _K = 40
    elif preset == 'M':
        _I = 112
        _J = 112
        _K = 80
    elif preset == 'L':
        _I = 180
        _J = 180
        _K = 160
    elif preset == 'paper':
        _I = 256
        _J = 256
        _K = 160

    sdfg = dace_module.vadv.to_sdfg(simplify=False)

    symbols = dict(I=_I, J=_J, K=_K)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = base_module.initialize(_I, _J, _K)
    if validate:
        dtr_stage_copy = np.copy(dtr_stage)
        utens_stage_copy = np.copy(utens_stage)
        u_stage_copy = np.copy(u_stage)
        wcon_copy = np.copy(wcon)
        u_pos_copy = np.copy(u_pos)
        utens_copy = np.copy(utens)

        numpy_module.vadv(utens_stage_copy, u_stage_copy, wcon_copy,
                          u_pos_copy, utens_copy, dtr_stage_copy)
        sdfg(dtr_stage=dtr_stage, utens_stage=utens_stage,
             u_stage=u_stage,
             wcon=wcon, u_pos=u_pos, utens=utens,
             **symbols)

        if not np.allclose(utens_stage_copy, utens_stage):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS,
                        dtr_stage=dtr_stage, utens_stage=utens_stage,
                        u_stage=u_stage,
                        wcon=wcon, u_pos=u_pos, utens=utens,
                        **symbols)
