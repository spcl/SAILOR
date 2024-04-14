import argparse
import dace
from npbench.benchmarks.polybench.doitgen import doitgen as base_module
from npbench.benchmarks.polybench.doitgen import doitgen_dace as dace_module

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

    increment=True

    if preset == 'S':
        _NR = 60
        _NQ = 60
        _NP = 128
    elif preset == 'M':
        _NR = 110
        _NQ = 125
        _NP = 256
    elif preset == 'L':
        _NR = 220
        _NQ = 250
        _NP = 512
    elif preset == 'paper':
        _NR = 220
        _NQ = 250
        _NP = 270

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(NR=_NR, NQ=_NQ, NP=_NP)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    A, C4 = base_module.initialize(_NR, _NQ, _NP)
    with dace.profile(repetitions=T_REPS):
        sdfg(A=A, C4=C4, **symbols)
