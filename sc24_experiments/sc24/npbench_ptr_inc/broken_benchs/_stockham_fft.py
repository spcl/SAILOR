import argparse
import dace
from npbench.benchmarks.stockham_fft import stockham_fft as base_module
from npbench.benchmarks.stockham_fft import stockham_fft_dace as dace_module

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
        _R = 2
        _K = 15
    elif preset == 'M':
        _R = 2
        _K = 18
    elif preset == 'L':
        _R = 2
        _K = 21
    elif preset == 'paper':
        _R = 4
        _K = 10

    sdfg = dace_module.stockham_fft.to_sdfg(simplify=False)

    symbols = dict(R=_R, K=_K)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    N, X, Y = base_module.initialize(_R, _K)
    with dace.profile(repetitions=T_REPS):
        sdfg(N=N, X=X, Y=Y, **symbols)
