import argparse
import dace
from npbench.benchmarks.deep_learning.lenet import lenet as base_module
from npbench.benchmarks.deep_learning.lenet import lenet_dace as dace_module

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
        _N = 4
        _H = 28
        _W = 28
    elif preset == 'M':
        _N = 8
        _H = 56
        _W = 56
    elif preset == 'L':
        _N = 8
        _H = 176
        _W = 176
    elif preset == 'paper':
        _N = 16
        _H = 256
        _W = 256

    sdfg = dace_module.lenet5.to_sdfg(simplify=False)

    symbols = dict(N=_N, W=_W, H=_H)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    (input, conv1, conv1bias, conv2, conv2bias, fc1w,
     fc1b, fc2w, fc2b, fc3w, fc3b,
     C_before_fc1) = base_module.initialize(_N, _H, _W)
    with dace.profile(repetitions=T_REPS):
        sdfg(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w,
             fc2b, fc3w, fc3b, C_before_fc1=C_before_fc1, **symbols)
