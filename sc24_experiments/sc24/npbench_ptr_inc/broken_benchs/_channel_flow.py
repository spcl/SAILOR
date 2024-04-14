import argparse
import dace
from npbench.benchmarks.channel_flow import channel_flow as base_module
from npbench.benchmarks.channel_flow import channel_flow_dace as dace_module

from sc24.npbench_ptr_inc import utils

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-r' '--repetitions', dest='repetitions', type=int, default=1000)
    argp.add_argument('-p' '--preset', dest='preset', type=str, default='S')
    argp.add_argument('--increment', dest='increment', default=False, action='store_true')
    argp.add_argument('--parallel', dest='parallel', default=False, action='store_true')
    args = argp.parse_args()

    T_REPS = args.repetitions

    increment = args.increment
    parallel = args.parallel
    preset = args.preset

    if preset == 'S':
        _nx = 61
        _ny = 61
        _nit = 5
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'M':
        _nx = 121
        _ny = 121
        _nit = 10
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'L':
        _nx = 201
        _ny = 201
        _nit = 20
        _rho = 1.0
        _nu = 0.1
        _F = 1.0
    elif preset == 'paper':
        _nx = 101
        _ny = 101
        _nit = 50
        _rho = 1.0
        _nu = 0.1
        _F = 1.0

    sdfg = dace_module.channel_flow.to_sdfg(simplify=False)

    symbols = dict(nx=_nx, ny=_ny, nit=_nit, rho=_rho, nu=_nu, F=_F)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    u, v, p, dx, dy, dt = base_module.initialize(_nx, _ny)
    with dace.profile(repetitions=T_REPS):
        sdfg(u=u, v=v, p=p, dx=dx, dy=dy, dt=dt, **symbols)
