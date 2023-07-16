#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>
#include <random>

#include "gpu_func.h"
//...
const int M = 200;
const int N = 100;

real random_real() {
    return (real)(rand()) / (real)(rand());
}
void test_gpu_add();
void test_l2norm();

int main() {
    test_gpu_add();
    test_l2norm();
}

void test_gpu_add() {
    std::cout << "test gpu_add: ";
    real hA[M * N];
    real hB[M * N];
    real hC[M * N];
    real hC_from_device[M * N];
    real *dA, *dB, *dC;
    cudaMalloc((void **)&dA, sizeof(real) * M * N);
    cudaMalloc((void **)&dB, sizeof(real) * M * N);
    cudaMalloc((void **)&dC, sizeof(real) * M * N);

    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(0, 100);

    real alpha = 1.0f, beta = 1.0f;
    for (int i = 0; i < M * N; ++i) {
        hA[i] = distr(eng);
        hB[i] = distr(eng);
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
        std::cout << "test passed\n";
    } else {
        std::cout << "failed, error: " << error / (M * N) << std::endl;
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void test_l2norm() {
    real hA[M * N];
    real *dA;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(0, 1);
    std::cout << "test l2norm: ";
    real host_norm = 0.0f;
    for (int i = 0; i < M*N; ++i) {
        hA[i] = distr(eng);
        host_norm += hA[i] * hA[i];
    }
    host_norm = sqrt(host_norm);

    cudaMalloc((void **)&dA, sizeof(real) * M * N);
    cudaMemcpy(dA, hA, sizeof(real) * M * N, cudaMemcpyHostToDevice);
    real device_norm = l2norm(dA, M, N);    
    cudaFree(dA);

    if (abs(host_norm - device_norm) >= 0.01) {
        std::cout << "failure, error: " << host_norm << " " << device_norm << std::endl;
    } else {
        std::cout << "passed.\n";
    }
}