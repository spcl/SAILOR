#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define BET_M 0.5
#define BET_P 0.5

static inline int int_compare(const void *a, const void *b) {
    int int_a = *((int *) a);
    int int_b = *((int *) b);
    return (int_a > int_b) - (int_a < int_b);
}

static inline int double_compare(const void *a, const void *b) {
    double d_a = *((double *) a);
    double d_b = *((double *) b);
    return ((d_a > d_b) ? 1 : ((d_a < d_b) ? -1 : 0));
}

double startT, endT;

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf("Error return from gettimeofday: %d\n", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

double randrange(double minval, double maxval) {
    double range = maxval - minval;
    double div = RAND_MAX / range;
    return minval + (rand() / div);
}

void init_random_array_3d(int x, int y, int z, double (*arr)[x][y][z]) {
    for (long long i = 0; i < x; i++)
        for (long long j = 0; j < y; j++)
            for (long long k = 0; k < z; k++)
                (*arr)[i][j][k] = randrange(0.0, 42.0);
}

void print_array(int x, int y, int z, double (*arr)[x][y][z]) {
    long long every = 0;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                every++;
                if (every % 1000 == 0)
                    fprintf(stderr, "%f\t", (*arr)[i][j][k]);
                if (every % 10000 == 0)
                    fprintf(stderr, "\n");
            }
        }
    }
    fprintf(stderr, "\n---------------------\n");
}

int check_array_equiv(
    int x, int y, int z, double (*arr_a)[x][y][z], double (*arr_b)[x][y][z]
) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                if (fabs((*arr_a)[i][j][k] - (*arr_b)[i][j][k]) > 1e-06)
                    return 0;
            }
        }
    }
    return 1;
}

void copy_array(
    int x, int y, int z, double (*arr_a)[x][y][z], double (*arr_b)[x][y][z]
) {
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            for (int k = 0; k < z; k++)
                (*arr_a)[i][j][k] = (*arr_b)[i][j][k];
}

void vadv_polly_helped(
    int I, int J, int K,
    double (* __restrict__ utens_stage)[I][J][K],
    double (* __restrict__ u_stage)[I][J][K],
    double (* __restrict__ wcon)[I + 1][J][K],
    double (* __restrict__ u_pos)[I][J][K],
    double (* __restrict__ utens)[I][J][K],
    double dtr_stage,
    double (* __restrict__ gcv)[I][J],
    double (* __restrict__ gav)[I][J],
    double (* __restrict__ cs)[I][J],
    double (* __restrict__ acol)[I][J],
    double (* __restrict__ bcol)[I][J],
    double (* __restrict__ ccol)[I][J][K],
    double (* __restrict__ dcol)[I][J][K],
    double (* __restrict__ correction_term)[I][J],
    double (* __restrict__ divided)[I][J],
    double (* __restrict__ as)[I][J],
    double (* __restrict__ data_col)[I][J],
    double (* __restrict__ datacol)[I][J]
) {

    int i, j, k;

#pragma scop
    for (k = 0; k < 1; k++) {
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*gcv)[i][j] = 0.25 * ((*wcon)[i + 1][j][k + 1] + (*wcon)[i][j][k + 1]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*cs)[i][j] = BET_M * (*gcv)[i][j];
                (*ccol)[i][j][k] = BET_P * (*gcv)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*bcol)[i][j] = dtr_stage - (*ccol)[i][j][k];
                (*correction_term)[i][j] = -(*cs)[i][j] * ((*u_stage)[i][j][k + 1] - (*u_stage)[i][j][k]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*dcol)[i][j][k] = (dtr_stage * (*u_pos)[i][j][k] + (*utens)[i][j][k] + (*utens_stage)[i][j][k] + (*correction_term)[i][j]);
            }
        }
        /* Thomas forward */
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*divided)[i][j] = 1.0 / (*bcol)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*ccol)[i][j][k] = (*divided)[i][j] * (*ccol)[i][j][k];
                (*dcol)[i][j][k] = (*divided)[i][j] * (*dcol)[i][j][k];
            }
        }
    }

    for (k = 1; k < K - 1; k++) {
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*gav)[i][j] = -0.25 * ((*wcon)[i + 1][j][k] + (*wcon)[i][j][k]);
                (*gcv)[i][j] = 0.25 * ((*wcon)[i + 1][j][k + 1] + (*wcon)[i][j][k + 1]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*as)[i][j] = BET_M * (*gav)[i][j];
                (*cs)[i][j] = BET_M * (*gcv)[i][j];
                (*acol)[i][j] = BET_P * (*gav)[i][j];
                (*ccol)[i][j][k] = BET_P * (*gcv)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*bcol)[i][j] = dtr_stage - (*acol)[i][j] - (*ccol)[i][j][k];
                (*correction_term)[i][j] = -(*as)[i][j] * ((*u_stage)[i][j][k - 1] - (*u_stage)[i][j][k]) - (*cs)[i][j] * ((*u_stage)[i][j][k + 1] - (*u_stage)[i][j][k]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*dcol)[i][j][k] = (dtr_stage * (*u_pos)[i][j][k] + (*utens)[i][j][k] + (*utens_stage)[i][j][k] + (*correction_term)[i][j]);
            }
        }
        /* Thomas forward */
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*divided)[i][j] = 1.0 / ((*bcol)[i][j] - (*ccol)[i][j][k - 1] * (*acol)[i][j]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*ccol)[i][j][k] = (*divided)[i][j] * (*ccol)[i][j][k];
                (*dcol)[i][j][k] = (*divided)[i][j] * ((*dcol)[i][j][k] - ((*dcol)[i][j][k - 1] * (*acol)[i][j]));
            }
        }
    }

    for (k = K - 1; k < K; k++) {
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*gav)[i][j] = -0.25 * ((*wcon)[i + 1][j][k] + (*wcon)[i][j][k]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*as)[i][j] = BET_M * (*gav)[i][j];
                (*acol)[i][j] = BET_P * (*gav)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*bcol)[i][j] = dtr_stage - (*acol)[i][j];
                (*correction_term)[i][j] = -(*as)[i][j] * ((*u_stage)[i][j][k - 1] - (*u_stage)[i][j][k]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*dcol)[i][j][k] = (dtr_stage * (*u_pos)[i][j][k] + (*utens)[i][j][k] + (*utens_stage)[i][j][k] + (*correction_term)[i][j]);
            }
        }
        /* Thomas forward */
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*divided)[i][j] = 1.0 / ((*bcol)[i][j] - (*ccol)[i][j][k - 1] * (*acol)[i][j]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*dcol)[i][j][k] = (*divided)[i][j] * ((*dcol)[i][j][k] - ((*dcol)[i][j][k - 1] * (*acol)[i][j]));
            }
        }
    }

    for (k = K - 1; k > K - 2; k -= 1) {
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*datacol)[i][j] = (*dcol)[i][j][k];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*data_col)[i][j] = (*datacol)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*utens_stage)[i][j][k] = dtr_stage * ((*datacol)[i][j] - (*u_pos)[i][j][k]);
            }
        }
    }

    for (k = K - 2; k > -1; k -= 1) {
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*datacol)[i][j] = ((*dcol)[i][j][k] - (*ccol)[i][j][k] * (*data_col)[i][j]);
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*data_col)[i][j] = (*datacol)[i][j];
            }
        }
        for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
                (*utens_stage)[i][j][k] = dtr_stage * ((*datacol)[i][j] - (*u_pos)[i][j][k]);
            }
        }
    }
#pragma endscop
}

double stddev(int N, double *data) {
    double sum = 0.0;
    double std = 0.0;
    int i;
    for (i = 0; i < N; ++i)
        sum += data[i];
    double mean = sum / N;
    for (i = 0; i < N; ++i)
        std += pow(data[i] - mean, 2);
    return sqrt(std / N);
}

int main(int argc, char **argv) {
    srand(time(NULL));

    int I = atoi(argv[1]);
    int J = atoi(argv[2]);
    int K = atoi(argv[3]);
    int T_REPS = atoi(argv[4]);

    double times[T_REPS];

    for (int t = 0; t < T_REPS; t++) {
        fprintf(stderr, "Running iteration %d/%d...\n", (t + 1), T_REPS);

        double (* restrict gcv)[I][J];
        double (* restrict gav)[I][J];
        double (* restrict cs)[I][J];
        double (* restrict acol)[I][J];
        double (* restrict bcol)[I][J];
        double (* restrict ccol)[I][J][K];
        double (* restrict dcol)[I][J][K];
        double (* restrict correction_term)[I][J];
        double (* restrict divided)[I][J];
        double (* restrict as)[I][J];
        double (* restrict data_col)[I][J];
        double (* restrict datacol)[I][J];

        double (* restrict utens_stage)[I][J][K];
        double (* restrict u_stage)[I][J][K];
        double (* restrict wcon)[I + 1][J][K];
        double (* restrict u_pos)[I][J][K];
        double (* restrict utens)[I][J][K];
        double dtr_stage;

        gcv = malloc(sizeof *gcv);
        gav = malloc(sizeof *gav);
        cs = malloc(sizeof *cs);
        acol = malloc(sizeof *acol);
        bcol = malloc(sizeof *bcol);
        ccol = malloc(sizeof *ccol);
        dcol = malloc(sizeof *dcol);
        correction_term = malloc(sizeof *correction_term);
        divided = malloc(sizeof *divided);
        as = malloc(sizeof *as);
        data_col = malloc(sizeof *data_col);
        datacol = malloc(sizeof *datacol);

        utens_stage = malloc(sizeof *utens_stage);
        u_stage = malloc(sizeof *u_stage);
        wcon = malloc(sizeof *wcon);
        u_pos = malloc(sizeof *u_pos);
        utens = malloc(sizeof *utens);

        init_random_array_3d(I, J, K, utens_stage);
        init_random_array_3d(I, J, K, u_stage);
        init_random_array_3d(I + 1, J, K, wcon);
        init_random_array_3d(I, J, K, u_pos);
        init_random_array_3d(I, J, K, utens);
        dtr_stage = 3.0 / 20.0;

        startT = rtclock();
        vadv_polly_helped(
            I, J, K,
            utens_stage,
            u_stage,
            wcon,
            u_pos,
            utens,
            dtr_stage,
            gcv,
            gav,
            cs,
            acol,
            bcol,
            ccol,
            dcol,
            correction_term,
            divided,
            as,
            data_col,
            datacol
        );
        endT = rtclock();
        times[t] = endT - startT;

        print_array(I, J, K, utens_stage);

        free(gcv);
        free(gav);
        free(cs);
        free(acol);
        free(bcol);
        free(ccol);
        free(dcol);
        free(correction_term);
        free(divided);
        free(as);
        free(data_col);
        free(datacol);

        free(utens_stage);
        free(u_stage);
        free(wcon);
        free(u_pos);
        free(utens);
    }

    double sum_pure = 0;
    for (int i = 0; i < T_REPS; i++)
        sum_pure += times[i];
    double avg_pure = sum_pure / T_REPS;

    qsort(&(times)[0], T_REPS, sizeof(double), double_compare);
    double median_reg = times[T_REPS / 2];

    fprintf(stderr, "Baseline:\t\t\t%f seconds (mean), %f seconds (median)\n", avg_pure, median_reg);
    fprintf(stdout, "Median: %f ms, Stddev: %f\n", median_reg * 1000, stddev(T_REPS, times));
    return 0;
}
