import argparse
import dace
from npbench.benchmarks.nbody import nbody as base_module
from npbench.benchmarks.nbody import nbody_dace as dace_module

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
        _N = 25
        _tEnd = 2.0
        _dt = 0.05
        _softening = 0.1
        _G = 1.0
    elif preset == 'M':
        _N = 50
        _tEnd = 5.0
        _dt = 0.02
        _softening = 0.1
        _G = 1.0
    elif preset == 'L':
        _N = 100
        _tEnd = 9.0
        _dt = 0.01
        _softening = 0.1
        _G = 1.0
    elif preset == 'paper':
        _N = 100
        _tEnd = 10.0
        _dt = 0.01
        _softening = 0.1
        _G = 1.0

    sdfg = dace_module.nbody.to_sdfg(simplify=False)

    symbols = dict(N=_N, tEnd=_tEnd, dt=_dt, softening=_softening, G=_G)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    mass, pos, vel, Nt = base_module.initialize(_N, _tEnd, _dt)
    with dace.profile(repetitions=T_REPS):
        sdfg(mass, pos, vel, Nt, **symbols)
