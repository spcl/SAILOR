import argparse
import numpy as np
import dace as dc
from dace.transformation.auto import auto_optimize

# Sample constants
BET_M = 0.5
BET_P = 0.5

dtype = dc.float64
I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))

# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@dc.program
def vadv(utens_stage: dtype[I, J, K], u_stage: dtype[I, J, K],
         wcon: dtype[I + 1, J, K], u_pos: dtype[I, J, K],
         utens: dtype[I, J, K], dtr_stage: dc.float64):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        # cs = gcv * BET_M
        cs[:] = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        # bcol = dtr_stage - acol - ccol[:, :, k]
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] -
        correction_term[:] = -as_ * (
            u_stage[:, :, k - 1] -
            u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        # as_ = gav * BET_M
        as_[:] = gav * BET_M
        # acol = gav * BET_P
        acol[:] = gav * BET_P
        # bcol = dtr_stage - acol
        bcol[:] = dtr_stage - acol

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        # datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('I')
    argp.add_argument('J')
    argp.add_argument('K')
    argp.add_argument('T_REPS')
    args = argp.parse_args()

    _I = int(args.I)
    _J = int(args.J)
    _K = int(args.K)

    T_REPS = int(args.T_REPS)

    rng = np.random.default_rng(42)
    dtr_stage = 3. / 20.
    utens_stage = rng.random((_I, _J, _K))
    u_stage = rng.random((_I, _J, _K))
    wcon = rng.random((_I + 1, _J, _K))
    u_pos = rng.random((_I, _J, _K))
    utens = rng.random((_I, _J, _K))

    sdfg = vadv.to_sdfg()
    sdfg.name = 'auto_optimized'

    auto_optimize.auto_optimize(sdfg, dc.DeviceType.CPU, symbols=dict(I=_I, J=_J, K=_K))
    sdfg.simplify()
    sdfg.save('dace_autoopt.sdfg')

    print('After auto-opt')
    with dc.profile(repetitions=T_REPS) as profile:
        sdfg(utens_stage=utens_stage, u_stage=u_stage, wcon=wcon, u_pos=u_pos,
             utens=utens, dtr_stage=dtr_stage, I=_I, J=_J, K=_K)

    report = profile.report
    alltimes = list(report.durations[(0, -1, -1)].values())[0][-1]
    print('Median:', np.median(alltimes), 'ms, Stddev:', np.std(alltimes))
