import argparse
import dace
from npbench.benchmarks.polybench.symm import symm as base_module
from npbench.benchmarks.polybench.symm import symm_dace as dace_module

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
        _M = 40
        _N = 50
    elif preset == 'M':
        _M = 120
        _N = 150
    elif preset == 'L':
        _M = 350
        _N = 550
    elif preset == 'paper':
        _M = 1000
        _N = 1200

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N, M=_M)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, C, A, B = base_module.initialize(_M, _N)
    with dace.profile(repetitions=T_REPS):
        sdfg(A=A, C=C, B=B, alpha=alpha, beta=beta, **symbols)
