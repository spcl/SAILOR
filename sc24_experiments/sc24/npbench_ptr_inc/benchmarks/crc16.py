import argparse
import numpy as np
from npbench.benchmarks.crc16 import crc16 as base_module
from npbench.benchmarks.crc16 import crc16_numpy as numpy_module
from npbench.benchmarks.crc16 import crc16_dace as dace_module

from sc24.npbench_ptr_inc import utils

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-r' '--repetitions', dest='repetitions', type=int, default=1000)
    argp.add_argument('-p' '--preset', dest='preset', type=str, default='S')
    argp.add_argument('--increment', dest='increment', default=False, action='store_true')
    argp.add_argument('--parallel', dest='parallel', default=False, action='store_true')
    argp.add_argument('--validate', dest='validate', default=False, action='store_true')
    args = argp.parse_args()

    T_REPS = args.repetitions

    increment = args.increment
    parallel = args.parallel
    preset = args.preset
    validate = args.validate

    if preset == 'S':
        _N = 1600
    elif preset == 'M':
        _N = 16000
    elif preset == 'L':
        _N = 160000
    elif preset == 'paper':
        _N = 1000000

    sdfg = dace_module.crc16.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    data = base_module.initialize(_N)
    if validate:
        data_copy = np.copy(data)

        ret_validate = numpy_module.crc16(data_copy)
        ret = sdfg(data=data, **symbols)

        if not np.allclose(ret, ret_validate):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, data=data, **symbols)
