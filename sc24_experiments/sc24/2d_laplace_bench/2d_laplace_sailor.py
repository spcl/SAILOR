import dace
from dace import dtypes
import numpy as np
from dace.transformation.auto import auto_optimize

I = dace.symbol('I')
J = dace.symbol('J')

@dace.program
def laplace_2d_sailor(in_field: dace.float64[I, J], out_field: dace.float64[I, J]):
    for j, i in dace.map[1:J-1, 1:I-1]:
        out_field[i,j] = 4.0 * in_field[i,j] - (
            in_field[i+1,j] + in_field[i-1,j] +
            in_field[i,j+1] + in_field[i,j-1])


if __name__ == '__main__':
    I = 4096
    J = 4096
    T_REPS = 100

    sdfg = laplace_2d_sailor.to_sdfg()
    auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU, symbols=dict(I=I, J=J))

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.map.schedule = dtypes.ScheduleType.Sequential

    for state in sdfg.states():
        for e in state.edges():
            if isinstance(e.src, dace.nodes.MapEntry) and isinstance(e.dst, dace.nodes.Tasklet):
                e.data.schedule = dtypes.MemletScheduleType.Pointer_Increment
            elif isinstance(e.dst, dace.nodes.MapExit) and isinstance(e.src, dace.nodes.Tasklet):
                e.data.schedule = dtypes.MemletScheduleType.Pointer_Increment

    in_field = np.random.rand(I,J)
    out_field = np.zeros((I,J), dtype=np.float64)

    with dace.profile(repetitions=T_REPS) as profiler:
        sdfg(in_field, out_field, I=I, J=J)

    print(profiler.report)
