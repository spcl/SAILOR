import argparse
import numpy as np
from npbench.benchmarks.cavity_flow import cavity_flow as base_module
from npbench.benchmarks.cavity_flow import cavity_flow_numpy as numpy_module
from npbench.benchmarks.cavity_flow import cavity_flow_dace as dace_module

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
        _nx = 61
        _ny = 61
        _nt = 25
        _nit = 5
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'M':
        _nx = 121
        _ny = 121
        _nt = 50
        _nit = 5
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'L':
        _nx = 201
        _ny = 201
        _nt = 100
        _nit = 20
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'paper':
        _nx = 101
        _ny = 101
        _nt = 700
        _nit = 50
        _rho = 1.0
        _nu = 0.1
        _F = 1.0

    sdfg = dace_module.cavity_flow.to_sdfg(simplify=False)

    symbols = dict(nx=_nx, ny=_ny, nt=_nt, nit=_nit, rho=_rho, nu=_nu, F=_F)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    u, v, p, dx, dy, dt = base_module.initialize(_nx, _ny)
    if validate:
        u_copy = np.copy(u)
        v_copy = np.copy(v)
        p_copy = np.copy(p)
        dx_copy = np.copy(dx)
        dy_copy = np.copy(dy)
        dt_copy = np.copy(dt)

        numpy_module.cavity_flow(_nx, _ny, _nt, _nit, u_copy, v_copy, dt_copy, dx_copy, dy_copy, p_copy, _rho, _nu)
        sdfg(u=u, v=v, p=p, dx=dx, dy=dy, dt=dt, **symbols)

        if not np.allclose(u, u_copy) or not np.allclose(p, p_copy) or not np.allclose(v, v_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, u=u, v=v, p=p, dx=dx, dy=dy, dt=dt, **symbols)
