#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>
#include <random>

#include "gpu_func.h"
//..
const int M = 200;
const int N = 100;

real random_real() {
    return (real)(rand()) / (real)(rand());
}
void test_gpu_add();


int main() {
    test_gpu_add();

}

void test_gpu_add() {
    real hA[M * N];
    real hB[M * N];
    real hC[M * N];
    real hC_from_device[M * N];
    real *dA, *dB, *dC;
    cudaMalloc((void **)&dA, sizeof(real) * M * N);
    cudaMalloc((void **)&dB, sizeof(real) * M * N);
    cudaMalloc((void **)&dC, sizeof(real) * M * N);

    real alpha = 1.0f, beta = 1.0f;
    for (int i = 0; i < M * N; ++i) {
        hA[i] = random_real();
        hB[i] = random_real();
        hC[i] = hA[i] * alpha + hB[i] * beta;
    }
    cudaMemcpy(dA, hA, sizeof(real) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(real) * M * N, cudaMemcpyHostToDevice);
    gpu_add(dA, dB, dC, alpha, beta, M, N);
    cudaMemcpy(hC_from_device, dC, sizeof(real) * M * N,
               cudaMemcpyDeviceToHost);
    // test
    bool pass = true;
    real error = 0;
    for (int i = 0; i < M * N; ++i) {
        if (hC[i] - hC_from_device[i] >= 0.001) {
            pass = false;
        }
        error += abs(hC[i] - hC_from_device[i]);
    }
    if (pass == true) {
        std::cout << "test passed for gpu_add\n";
    } else {
        std::cout << "failed, error: " << error / (M * N) << std::endl;
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}