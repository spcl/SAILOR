import argparse
import dace
from npbench.benchmarks.spmv import spmv as base_module
from npbench.benchmarks.spmv import spmv_dace as dace_module

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
        _M = 4096
        _N = 4096
        _nnz = 8192
    elif preset == 'M':
        _M = 32768
        _N = 32768
        _nnz = 65536
    elif preset == 'L':
        _M = 262144
        _N = 262144
        _nnz = 262144
    elif preset == 'paper':
        _M = 131072
        _N = 1131072
        _nnz = 262144

    sdfg = dace_module.spmv.to_sdfg(simplify=False)

    symbols = dict(M=_M, N=_N, nnz=_nnz)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    rows, cols, vals, x = base_module.initialize(_M, _N, _nnz)
    with dace.profile(repetitions=T_REPS):
        sdfg(A_row=rows, A_col=cols, A_val=vals, x=x, **symbols)
