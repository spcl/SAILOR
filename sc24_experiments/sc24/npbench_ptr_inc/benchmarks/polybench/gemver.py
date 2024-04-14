import argparse
import numpy as np
from npbench.benchmarks.polybench.gemver import gemver as base_module
from npbench.benchmarks.polybench.gemver import gemver_numpy as numpy_module
from npbench.benchmarks.polybench.gemver import gemver_dace as dace_module

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
    validate= args.validate

    if preset == 'S':
        _N = 1000
    elif preset == 'M':
        _N = 3000
    elif preset == 'L':
        _N = 10000
    elif preset == 'paper':
        _N = 8000

    sdfg = dace_module.kernel.to_sdfg(simplify=False)

    symbols = dict(N=_N)

    utils.prepare_sdfg(sdfg, parallel, increment, symbols)

    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = base_module.initialize(_N)
    if validate:
        alpha_copy = np.copy(alpha)
        beta_copy = np.copy(beta)
        A_copy = np.copy(A)
        u1_copy = np.copy(u1)
        v1_copy = np.copy(v1)
        u2_copy = np.copy(u2)
        v2_copy = np.copy(v2)
        w_copy = np.copy(w)
        x_copy = np.copy(x)
        y_copy = np.copy(y)
        z_copy = np.copy(z)

        numpy_module.kernel(alpha_copy, beta_copy, A_copy, u1_copy,
                            v1_copy, u2_copy, v2_copy, w_copy, x_copy, y_copy,
                            z_copy)
        sdfg(alpha=alpha, beta=beta, A=A, u1=u1, u2=u2, v1=v1, v2=v2,
            w=w, x=x, y=y, z=z, **symbols)

        if not np.allclose(A_copy, A) or not np.allclose(x, x_copy) or not np.allclose(w, w_copy):
            print('validation FAILED!')
        else:
            print('success')
    else:
        utils.benchmark(sdfg, T_REPS, alpha=alpha, beta=beta, A=A, u1=u1,
                        u2=u2, v1=v1, v2=v2, w=w, x=x, y=y, z=z, **symbols)
