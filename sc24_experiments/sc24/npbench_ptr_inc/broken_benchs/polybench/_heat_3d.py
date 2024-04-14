import argparse
import dace
from npbench.benchmarks.polybench.heat_3d import heat_3d as base_module
from npbench.benchmarks.polybench.heat_3d import heat_3d_dace as dace_module

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
        _TSTEPS = 25
        _N = 25
    elif preset == 'M':
        _TSTEPS = 50
        _N = 40
    elif preset == 'L':
        _TSTEPS = 100
        _N = 70
    elif preset == 'paper':
        _TSTEPS = 500
        _N = 120

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, TSTEPS=_TSTEPS)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A, B = base_module.initialize(_N)
    with dace.profile(repetitions=T_REPS):
        sdfg(A=A, B=B, **symbols)
