#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

void laplace_2d(int I, int J, double *lap, double *in,
    int lap_sI, int lap_sJ, int in_sI, int in_sJ) {
#pragma scop
for (int j=1; j < J-1; ++j) {
  for (int i=1; i < I-1; ++i) {
    lap[i*lap_sI + j*lap_sJ] = 4.0 * in[i*in_sI + j*in_sJ] - (
      in[(i+1)*in_sI + j*in_sJ] + in[(i-1)*in_sI + j*in_sJ] +
      in[i*in_sI + (j+1)*in_sJ] + in[i*in_sI + (j-1)*in_sJ]
    );
  }
}
#pragma endscop
}

double randrange(double minval, double maxval) {
    double range = maxval - minval;
    double div = RAND_MAX / range;
    return minval + (std::rand() / div);
}

void setzero_array_2d(int x, int y, int sx, int sy, double *arr) {
    for (long long i = 0; i < x; i++)
        for (long long j = 0; j < y; j++)
            arr[i * sx + j * sy] = 0.0;
}

void init_random_array_2d(int x, int y, int sx, int sy, double *arr) {
    for (long long i = 0; i < x; i++)
        for (long long j = 0; j < y; j++)
            arr[i * sx + j * sy] = randrange(0.0, 42.0);
}

void print_array_2d(int x, int y, int sx, int sy, double *arr) {
    for (long long i = 0; i < x; i++) {
        for (long long j = 0; j < y; j++) {
            if (i % 100 == 0)
                fprintf(stderr, "%.2f, ", arr[i * sx + j * sy]);
        }
    }
}

int main(int argc, char **argv) {
    std::srand(std::time(NULL));

    int I = std::atoi(argv[1]);
    int J = std::atoi(argv[2]);
    int sI1 = std::atoi(argv[3]);
    int sJ1 = std::atoi(argv[4]);
    int sI2 = std::atoi(argv[5]);
    int sJ2 = std::atoi(argv[6]);
    int T_REPS = std::atoi(argv[7]);

    double *A_orig;
    double *B_orig;

    std::vector<double> times_orig(T_REPS);

    for (int i = 0; i < T_REPS; i++) {
        std::cout << "iteration " << i << "/" << T_REPS << std::endl;

        A_orig = (double*) malloc(sizeof(double) * I * J);
        B_orig = (double*) malloc(sizeof(double) * I * J);

        init_random_array_2d(I, J, sI1, sJ1, A_orig);
        setzero_array_2d(I, J, sI2, sJ2, B_orig);

        auto s1 = std::chrono::high_resolution_clock::now();
        laplace_2d(I, J, B_orig, A_orig, sI1, sJ1, sI2, sJ2);
        auto s2 = std::chrono::high_resolution_clock::now();

        times_orig[i] = std::chrono::duration_cast<std::chrono::microseconds>(s2 - s1).count() / 1000.0;

        print_array_2d(I, J, sI2, sJ2, B_orig);

        free(A_orig);
        free(B_orig);
    }

    std::sort(times_orig.begin(), times_orig.end());

    std::cout << "Median original: " << times_orig[std::floor(T_REPS / 2)] << " ms" << std::endl;

    return 0;
}


