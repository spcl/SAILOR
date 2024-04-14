import argparse
import dace
from npbench.benchmarks.contour_integral import contour_integral as base_module
from npbench.benchmarks.contour_integral import contour_integral_dace as dace_module

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
        _NR = 50
        _NM = 150
        _slab_per_bc = 2
        _num_int_pts = 32
    elif preset == 'M':
        _NR = 200
        _NM = 400
        _slab_per_bc = 2
        _num_int_pts = 32
    elif preset == 'L':
        _NR = 600
        _NM = 1000
        _slab_per_bc = 2
        _num_int_pts = 32
    elif preset == 'paper':
        _NR = 500
        _NM = 1000
        _slab_per_bc = 2
        _num_int_pts = 32

    sdfg = dace_module.contour_integral.to_sdfg(simplify=False)

    symbols = dict(NR=_NR, NM=_NM, slab_per_bc=_slab_per_bc, num_int_pts=_num_int_pts)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    Ham, int_pts, Y = base_module.initialize(_NR, _NM, _slab_per_bc, _num_int_pts)
    with dace.profile(repetitions=T_REPS):
        sdfg(Ham=Ham, int_pts=int_pts, Y=Y, **symbols)
