import argparse
import sys
import numpy as np
import dace
from dace.transformation.auto import auto_optimize
from sc24.vadv.vadv_dace import vadv


T_REPS = 500

_I = 128
_J = 128
_K = 80

WITH_PTR_INCR = False
WITH_COLLAPSE = False
WITH_DOACROSS = False
WITH_SCHEDULE = 'dynamic'

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('I')
    argp.add_argument('J')
    argp.add_argument('K')
    argp.add_argument('T_REPS')
    argp.add_argument('--doacross', dest='doacross', default=False, action='store_true')
    argp.add_argument('--collapse', dest='collapse', default=False, action='store_true')
    argp.add_argument('--increment', dest='increment', default=False, action='store_true')
    argp.add_argument('--schedule', dest='schedule', default='dynamic')
    args = argp.parse_args()

    _I = int(args.I)
    _J = int(args.J)
    _K = int(args.K)
    T_REPS = int(args.T_REPS)

    WITH_COLLAPSE = args.collapse
    WITH_PTR_INCR = args.increment
    WITH_DOACROSS = args.doacross
    WITH_SCHEDULE = args.schedule


if __name__ == '__main__':
    if WITH_DOACROSS:
        vadv.use_experimental_cfg_blocks = True
        sdfg = vadv.to_sdfg(simplify=False)
        auto_optimize.auto_parallelize(sdfg, dace.DeviceType.CPU,
                                       symbols=dict(I=_I, J=_J, K=_K),
                                       use_doacross_parallelism=WITH_DOACROSS,
                                       force_collapse_parallelism=WITH_COLLAPSE,
                                       use_pointer_incrementation=WITH_PTR_INCR)
    else:
        vadv.use_experimental_cfg_blocks = False
        sdfg = vadv.to_sdfg(simplify=True)
        auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU,
                                    symbols=dict(I=_I, J=_J, K=_K),
                                    with_canonicalization=False)

    print('Compiling', file=sys.stderr)
    compiled_sdfg = sdfg.compile()

    print('Performing profiling', file=sys.stderr)
    rng = np.random.default_rng(42)
    dtr_stage = 3. / 20.
    utens_stage = rng.random((_I, _J, _K), dtype=np.float64)
    u_stage = rng.random((_I, _J, _K), dtype=np.float64)
    wcon = rng.random((_I + 1, _J, _K), dtype=np.float64)
    u_pos = rng.random((_I, _J, _K), dtype=np.float64)
    utens = rng.random((_I, _J, _K), dtype=np.float64)

    with dace.profile(repetitions=T_REPS) as profile:
        compiled_sdfg(utens_stage=utens_stage, u_stage=u_stage, wcon=wcon, u_pos=u_pos, utens=utens,
                        dtr_stage=dtr_stage, I=_I, J=_J, K=_K)

    report = profile.report
    alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
    print('Median:', np.median(alltimes), 'ms, Stddev:', np.std(alltimes))
