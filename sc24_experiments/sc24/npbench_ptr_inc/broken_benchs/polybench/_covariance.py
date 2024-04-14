import argparse
import dace
from npbench.benchmarks.polybench.correlation import correlation as base_module
from npbench.benchmarks.polybench.correlation import correlation_dace as dace_module

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
        _M = 500
        _N = 600
    elif preset == 'M':
        _M = 1400
        _N = 1800
    elif preset == 'L':
        _M = 3200
        _N = 4000
    elif preset == 'paper':
        _M = 1200
        _N = 1400

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    float_n, data = base_module.initialize(_M, _N)
    with dace.profile(repetitions=T_REPS):
        sdfg(float_n=float_n, data=data, **symbols)
