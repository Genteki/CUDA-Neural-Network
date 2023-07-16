#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../utils/types.h"

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int myGEMM(const real* __restrict__ A, const real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta, int M, int N, int K);

// TODO
// Add additional function declarations
__global__ void GEMM_kernel(const real* __restrict__ A,
                            const real* __restrict__ B, real* __restrict__ C,
                            real alpha, real beta, int M, int N, int K);

__global__ void gpu_sigmoid_kernel(const real* __restrict__ A,
                                   real* __restrict__ C, int M, int N);

int gpu_sigmoid(const real* __restrict__ A, real* __restrict__ C, int M, int N);

__global__ void gpu_softmax_kernel(const real* __restrict__ A,
                                   real* __restrict__ C, int M,
                                   int N);

int gpu_softmax(const real* __restrict__ A, real* __restrict__ C, int M, int N);

int gpu_linear(const real* __restrict__ W, const real* __restrict__ x,
               const real* __restrict__ b, real* __restrict__ z, int in_dim,
               int out_dim, int batch_size);

int gpu_add(const real* A, const real* B,
             real* C, const real& alpha, const real& beta,
             const int& M, const int& N);

int gpu_element_multiply(const real* __restrict__ A,
                          const real* __restrict__ B, real* __restrict__ C,
                          const real& alpha, const int& M, const int& N);


int transpose(const real* __restrict__ A, real* __restrict__ B, const int& M,
               const int& N);

int sum_row(const real* __restrict__ A, real* __restrict__ B, const int& M,
             const int& N);

int matrix_add_const(const real* src, real* dst, const int& c, const int& M,
                      const int& N);

real l2norm(const real* src, const int& M, const int& N);

#endif
