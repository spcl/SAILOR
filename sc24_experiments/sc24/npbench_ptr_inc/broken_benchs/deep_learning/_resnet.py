import argparse
import dace
from npbench.benchmarks.deep_learning.resnet import resnet as base_module
from npbench.benchmarks.deep_learning.resnet import resnet_dace as dace_module

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
        _W = 14
        _H = 14
        _C1 = 32
        _C2 = 8
    elif preset == 'M':
        _N = 8
        _W = 28
        _H = 28
        _C1 = 64
        _C2 = 16
    elif preset == 'L':
        _N = 8
        _W = 56
        _H = 56
        _C1 = 128
        _C2 = 32
    elif preset == 'paper':
        _N = 8
        _W = 56
        _H = 56
        _C1 = 256
        _C2 = 64

    sdfg = dace_module.resnet_basicblock.to_sdfg(simplify=False)

    symbols = dict(N=_N, W=_W, H=_H, C1=_C1, C2=_C2)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    (input, conv1, conv2, conv3) = base_module.initialize(_N, _W, _H, _C1, _C2)
    with dace.profile(repetitions=T_REPS):
        sdfg(input, conv1, conv2, conv3, **symbols)
