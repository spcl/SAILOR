import argparse
from tkinter import W
from typing import Tuple
import numpy as np
import dace

from dace import dtypes
from dace.sdfg import nodes as nd
from dace.transformation.dataflow import (DoubleBuffering, MapCollapse, MapExpansion, MapReduceFusion, StripMining,
                                          InLocalStorage, AccumulateTransient, Vectorization)
from dace.transformation import helpers as xfutil
from dace.transformation.auto import auto_optimize

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

K_TILE = 256
I_TILE = 32
J_TILE = 128
II_TILE = 4
JJ_TILE = 32

VECSIZE = 8

dtype = dace.float64
np_dtype = np.float64

@dace.program
def matmul_mkl(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N]):
    C[:] = A @ B

# Map-Reduce version of matrix multiplication
@dace.program
def matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_map_and_state_by_param(sdfg: dace.SDFG, pname: str) -> Tuple[dace.nodes.MapEntry, dace.SDFGState]:
    """ Finds the first map entry node by the given parameter name. """
    return next(
        (n, p) for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_mapexit_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapExit:
    """ Finds the first map exit node by the given parameter name. """
    entry, state = find_map_and_state_by_param(sdfg, pname)
    return state.exit_node(entry)


def param_group_orig(sdfg):
    K_TILE = 256
    I_TILE = 32
    J_TILE = 32
    II_TILE = 4
    JJ_TILE = 16

    VECSIZE = 4


def param_group_1(sdfg):
    K_TILE = 256
    I_TILE = 32
    J_TILE = 128
    II_TILE = 4
    JJ_TILE = 32

    VECSIZE = 8

    # Setup prefetching
    tile_j, state = find_map_and_state_by_param(sdfg, 'tile1_j')
    for oe in state.out_edges(tile_j):
        if oe.data.data == 'B':
            oe.data.schedule = dtypes.MemletScheduleType.Prefetch_Start
    for ie in state.in_edges(state.exit_node(tile_j)):
        if ie.data.data == 'C':
            ie.data.schedule = dtypes.MemletScheduleType.Prefetch_Start


def optimize_for_cpu(sdfg: dace.SDFG, m: int, n: int, k: int, target_compiler: str = 'clang'):
    """ Optimize the matrix multiplication example for multi-core CPUs. """
    # Ensure integers are 32-bit by default
    #dace.Config.set('compiler', 'default_data_types', value='C')

    if target_compiler == 'clang':
        # Best raw time: (Config A)
        #K_TILE = 256
        #I_TILE = 16
        #J_TILE = 512
        #II_TILE = 4
        #JJ_TILE = 32

        #VECSIZE = 16

        # Gives prefetch benefit: (Config B)
        #K_TILE = 256
        #I_TILE = 32
        #J_TILE = 32
        #II_TILE = 4
        #JJ_TILE = 8

        #VECSIZE = 4

        # DaCe version: (Config C)
        K_TILE = 256
        I_TILE = 32
        J_TILE = 32
        II_TILE = 4
        JJ_TILE = 16

        VECSIZE = 4
    else:
        K_TILE = 256
        I_TILE = 32
        J_TILE = 32
        II_TILE = 4
        JJ_TILE = 16

        VECSIZE = 8

    # Fuse the map and reduce nodes
    sdfg.apply_transformations(MapReduceFusion)

    # Find multiplication map
    entry = find_map_by_param(sdfg, 'k')

    # Create a tiling strategy
    divides_evenly = (m % I_TILE == 0) and (n % J_TILE == 0) and (k % K_TILE == 0)
    xfutil.tile(sdfg, entry, divides_evenly, False, k=K_TILE, i=I_TILE, j=J_TILE)
    xfutil.tile(sdfg, entry, divides_evenly, divides_evenly, j=JJ_TILE, i=II_TILE)

    # Reorder internal map to "k,i,j"
    xfutil.permute_map(entry, [2, 0, 1])

    # Add local storage for B in j tile: we apply InLocalStorage with a
    # parameter "array" named B, between the two maps of j and i
    regtile_j = find_map_by_param(sdfg, 'tile1_j')
    regtile_i = find_map_by_param(sdfg, 'tile1_i')
    InLocalStorage.apply_to(sdfg, dict(array='B'), node_a=regtile_j, node_b=regtile_i)

    if divides_evenly:
        # Add local storage for C
        exit_inner = find_mapexit_by_param(sdfg, 'k')
        exit_rti = find_mapexit_by_param(sdfg, 'tile1_i')
        AccumulateTransient.apply_to(sdfg, dict(array='C', identity=0), map_exit=exit_inner, outer_map_exit=exit_rti)

        # Vectorize microkernel map
        postamble = n % VECSIZE != 0
        entry_inner, inner_state = find_map_and_state_by_param(sdfg, 'k')
        Vectorization.apply_to(inner_state.parent,
                               dict(vector_len=VECSIZE, preamble=False, postamble=postamble),
                               map_entry=entry_inner)

    # Mark outer tile map as sequential to remove atomics
    find_map_by_param(sdfg, 'tile_k').map.schedule = dace.ScheduleType.Sequential

    # Collapse maps for more parallelism
    find_map_by_param(sdfg, 'o0').map.collapse = 2
    tile_i = find_map_by_param(sdfg, 'tile_i')
    tile_j = find_map_by_param(sdfg, 'tile_j')
    MapCollapse.apply_to(sdfg, outer_map_entry=tile_i, inner_map_entry=tile_j)
    tile_ij = find_map_by_param(sdfg, 'tile_i')  # Find newly created map
    tile_ij.map.schedule = dace.ScheduleType.CPU_Multicore
    tile_ij.map.collapse = 2


def do_ptr_increments(sdfg: dace.SDFG):
    k_map, state = find_map_and_state_by_param(sdfg, 'k')
    o_range = k_map.map.range[2]
    n_range = [o_range[0], 15, o_range[2]]
    k_map.map.range[2] = n_range
    for oe in state.out_edges(k_map):
        if isinstance(oe.dst, nd.Tasklet):
            oe.data.schedule = dtypes.MemletScheduleType.Pointer_Increment


def do_prefetch(sdfg: dace.SDFG):
    tile_j, state = find_map_and_state_by_param(sdfg, 'tile1_j')
    for oe in state.out_edges(tile_j):
        if oe.data.data == 'B':
            oe.data.schedule = dtypes.MemletScheduleType.Prefetch_Start
            oe.data.prefetch_locality = dtypes.MemletPrefetchType.High_Locality
    for ie in state.in_edges(state.exit_node(tile_j)):
        if ie.data.data == 'C':
            ie.data.schedule = dtypes.MemletScheduleType.Prefetch_Start
            ie.data.prefetch_locality = dtypes.MemletPrefetchType.High_Locality

    tile1_i, state = find_map_and_state_by_param(sdfg, 'tile1_i')
    for oe in state.out_edges(tile1_i):
        if oe.data.data == 'A':
            oe.data.schedule = dtypes.MemletScheduleType.Prefetch_Start
            oe.data.prefetch_locality = dtypes.MemletPrefetchType.High_Locality

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('N')
    argp.add_argument('M')
    argp.add_argument('K')
    argp.add_argument('T_REPS')
    argp.add_argument('--version', dest='version', default='opt')
    args = argp.parse_args()

    N = int(args.N)
    M = int(args.M)
    K = int(args.K)
    T_REPS = int(args.T_REPS)
    version = args.version

    #N = 1024
    #M = 1024
    #K = 1024
    #T_REPS = 100
    #version = 'opt'

    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.random.rand(M, N)

    if version == 'opt':
        opt_sdfg = matmul.to_sdfg()
        opt_sdfg.name = 'matmul_opt'
        optimize_for_cpu(opt_sdfg, M, N, K)
        do_prefetch(opt_sdfg)
        #do_ptr_increments(opt_sdfg)
        with dace.profile(repetitions=T_REPS) as profiler:
            opt_sdfg(A=A, B=B, C=C, N=N, M=M, K=K)

        print('Optimized:')
        report = profiler.report
        alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
        print(report)
        print('Median:', np.median(alltimes))
        print('Stddev:', np.std(alltimes))
    elif version == 'dace':
        dace_baseline_sdfg = matmul.to_sdfg()
        optimize_for_cpu(dace_baseline_sdfg, M, N, K)
        with dace.profile(repetitions=T_REPS) as profiler:
            dace_baseline_sdfg(A=A, B=B, C=C, N=N, M=M, K=K)

        print('DaCe Baseline:')
        report = profiler.report
        alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
        print(report)
        print('Median:', np.median(alltimes))
        print('Stddev:', np.std(alltimes))
    elif version == 'mkl':
        with dace.profile(repetitions=T_REPS) as profiler:
            matmul_mkl(A=A, B=B, C=C, N=N, M=M, K=K)

        print('MKL Baseline:')
        report = profiler.report
        alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
        print(report)
        print('Median:', np.median(alltimes))
        print('Stddev:', np.std(alltimes))
    elif version == 'dace-autoopt':
        dace_autoopt_sdfg = matmul.to_sdfg()
        dace_autoopt_sdfg.name = 'matmul_autoopt'
        auto_optimize.auto_optimize(dace_autoopt_sdfg, dtypes.DeviceType.CPU,
                                    symbols=dict(N=N, M=M, K=K),
                                    with_canonicalization=False,
                                    move_loop_into_maps=False)
        with dace.profile(repetitions=T_REPS) as profiler:
            dace_autoopt_sdfg(A=A, B=B, C=C, N=N, M=M, K=K)

        print('DaCe Autoopt:')
        report = profiler.report
        alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
        print(report)
        print('Median:', np.median(alltimes))
        print('Stddev:', np.std(alltimes))
    else:
        raise argparse.ArgumentError('Unknown version %s' % version)
