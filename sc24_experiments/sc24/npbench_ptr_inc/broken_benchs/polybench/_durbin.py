import argparse
import dace
from npbench.benchmarks.polybench.durbin import durbin as base_module
from npbench.benchmarks.polybench.durbin import durbin_dace as dace_module

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
        _N = 1000
    elif preset == 'M':
        _N = 6000
    elif preset == 'L':
        _N = 20000
    elif preset == 'paper':
        _N = 16000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    r = base_module.initialize(_N)
    with dace.profile(repetitions=T_REPS):
        sdfg(r=r, **symbols)
