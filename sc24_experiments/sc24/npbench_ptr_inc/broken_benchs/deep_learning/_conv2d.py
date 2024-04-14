import argparse
import dace
from npbench.benchmarks.deep_learning.conv2d_bias import conv2d as base_module
from npbench.benchmarks.deep_learning.conv2d_bias import conv2d_dace as dace_module

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
        _N = 8
        _C_in = 3
        _C_out = 16
        _K = 2
        _H = 32
        _W = 32
    elif preset == 'M':
        _N = 8
        _C_in = 3
        _C_out = 8
        _K = 5
        _H = 64
        _W = 64
    elif preset == 'L':
        _N = 8
        _C_in = 3
        _C_out = 8
        _K = 10
        _H = 128
        _W = 128
    elif preset == 'paper':
        _N = 8
        _C_in = 3
        _C_out = 16
        _K = 20
        _H = 256
        _W = 256

    sdfg = dace_module.conv2d_bias.to_sdfg(simplify=False)

    symbols = dict(N=_N, W=_W, C_in=_C_in, C_out=_C_out, H=_H, K=_K)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    input, weights, bias = base_module.initialize(_C_in, _C_out, _H, _K, _N, _W)
    with dace.profile(repetitions=T_REPS):
        sdfg(input=input, weights=weights, bias=bias, **symbols)
