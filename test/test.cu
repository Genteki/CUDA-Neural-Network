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
void test_sigmoid();
void test_rowsum();

int main() {
    test_gpu_add();
    test_l2norm();
    test_sigmoid();
    test_rowsum();
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

void test_sigmoid() {
    real h[M*N], hs[M*N];
    real *d_0, *d_s;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(-1, 1);
    std::cout << "test sigmoid: ";
    for (int i = 0; i < M*N; ++i) {
        h[i] = distr(eng);
    }
    cudaMalloc((void**)&d_0, sizeof(real)*M*N);
    cudaMalloc((void**)&d_s, sizeof(real)*M*N);
    cudaMemcpy(d_0, h, sizeof(real)*M*N, cudaMemcpyHostToDevice);
    gpu_sigmoid(d_0, d_s, M, N);
    cudaMemcpy(hs, d_s, sizeof(real)*M*N, cudaMemcpyDeviceToHost);
    bool pass = true;
    real error = 0.0f;
    for (int i = 0; i < M*N; ++i) {
        real sig = 1/ (exp(-h[i]) + 1);
        error += abs(sig - hs[i]);
        if (abs(sig - hs[i] >= 0.001)) {
            pass = false;
        }
    }
    if (pass) {
        std::cout << "passed\n";
    } else {
        std::cout << "failed, error: " << error << "\n";
    }
}

void test_rowsum() {
    real h_arr[M* N], h_sum[M], h_sum_from_d[M];
    real *d_arr, *d_sum;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(-1, 10);
    std::cout << "test row sum: ";
    for (int i = 0; i < M*N; ++i) {
        h_arr[i] = distr(eng);
    }
    cudaMalloc((void **)&d_arr, M * N * sizeof(real));
    cudaMalloc((void **)&d_sum, M * 1 * sizeof(real));
    cudaMemcpy(d_arr, h_arr, sizeof(real) * M * N, cudaMemcpyHostToDevice);
    memset(h_sum, 0, sizeof(real) * M);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col)
            h_sum[row] += h_arr[row + col * M];
    }
    sum_row(d_arr, d_sum, M, N);
    cudaMemcpy(h_sum_from_d, d_sum, sizeof(real) * M, cudaMemcpyDeviceToHost);

    bool pass = true;
    real error = 0.0f;
    for (int i = 0; i < M; ++i) {
        if (abs(h_sum[i] - h_sum_from_d[i]) >= 0.001) {
            pass = false;
            error += abs(h_sum - h_sum_from_d);
            std::cout << "error in row " << i << ": ref " << h_sum[i] << " dev " << h_sum_from_d[i] << std::endl;
        }
    }
    if (pass) {
        std::cout << "passed\n";
    } else {
        std::cout << "failed, error: " << h_sum[0] << h_sum_from_d[0] << "\n";
    }
}