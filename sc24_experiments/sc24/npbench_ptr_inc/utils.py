import numpy as np
import inspect
import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.auto import auto_optimize


def benchmark(sdfg: dace.SDFG, repetitions: int, **kwargs):
    with dace.profile(repetitions=repetitions) as profiler:
        sdfg(**kwargs)

    report = profiler.report
    alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
    #print(report)
    print('Median:', np.median(alltimes))
    print('Stddev:', np.std(alltimes))


def prepare_sdfg(sdfg: dace.SDFG, parallel: bool, increment: bool, symbols: dict = None, fuse_stencils: bool = True):
    caller_filename = inspect.stack()[1].filename
    sdfg_name = caller_filename.split('/')[-1].strip('.py')
    if parallel:
        sdfg_name += '_parallel'
    else:
        sdfg_name += '_sequential'
    if increment:
        sdfg_name += '_increment'
    else:
        sdfg_name += '_no_increment'
    sdfg.name = sdfg_name

    sdfg.simplify()
    sdfg.reset_cfg_list()

    auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU, symbols=symbols,
                                move_loop_into_maps=False,
                                with_canonicalization=False,
                                fuse_stencils=fuse_stencils,
                                use_fast_library_calls=True)
    sdfg.simplify()
    sdfg.reset_cfg_list()
    sdfg.validate()

    if not parallel:
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for nd in state.nodes():
                    if isinstance(nd, nodes.MapEntry):
                        nd.map.schedule = dtypes.ScheduleType.Sequential

    if increment:
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for edge in state.edges():
                    if (isinstance(edge.src, nodes.MapEntry) and isinstance(edge.dst, nodes.Tasklet) or
                        isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.MapExit)):
                        memlet: dace.Memlet = edge.data
                        if not memlet.data or not memlet.subset:
                            continue

                        desc = sd.data(memlet.data)
                        if not (isinstance(desc, dace.data.Array) and desc.total_size != 1):
                            continue

                        memlet.schedule = dtypes.MemletScheduleType.Pointer_Increment
