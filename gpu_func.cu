#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>

#include "cublas_v2.h"
#include "gpu_func.h"

#define BLOCK_DIM_X 16
#define BLOCK_SIZE 256
#define BLOCK_SIZE_1D 128
typedef unsigned int uint;

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

int myGEMM(const real* __restrict__ A, const real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta, int M, int N, int K) {
    dim3 bdim(BLOCK_DIM_X, BLOCK_SIZE / BLOCK_DIM_X);
    int grid_x = (N + bdim.x - 1) / bdim.x;
    int grid_y = (M + bdim.y - 1) / bdim.y;

    dim3 gdim(grid_x, grid_y);
    GEMM_kernel<<<gdim, bdim>>>(A, B, C, *alpha, *beta, M, N, K);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}

__global__ void GEMM_kernel(const real* __restrict__ A,
                            const real* __restrict__ B, real* __restrict__ C,
                            real alpha, real beta, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M && col < N) {
        real sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[row + M * i] * B[i + col * K];
        }
        int idx_c = row + col * M;
        C[idx_c] = sum * alpha + C[idx_c] * beta;
    }
    return;
}

/* Helper functions for neural networks */

__global__ void gpu_sigmoid_kernel(const real* __restrict__ A,
                                   real* __restrict__ C, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row + col * M;
    // sigmoid(x) = 1 / (1 + exp(-x))
    if (row < M && col < N) {
        C[idx] = 1 / ( 1 + expf(A[idx]));
    }
}

int gpu_sigmoid(const real* __restrict__ A, real* __restrict__ C, int M, int N=1) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_SIZE / BLOCK_DIM_X);
    int grid_x = (N + dim_block.x - 1) / dim_block.x;
    int grid_y = (M + dim_block.y - 1) / dim_block.y;
    dim3 dim_grid(grid_x, grid_y);
    gpu_sigmoid_kernel<<<dim_grid, dim_block>>>(A, C, M, N);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

__global__ void gpu_softmax_kernel(const real* __restrict__ A,
                                   real* __restrict__ C, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    real exp_sum;
    // softmax(xi) = exp(xi) / sum_i(exp(xi))
    if (col < N) {
        for (int row = 0; row < M; ++row) {
            int idx = row + col * M;
            real exp_xi = A[idx];
            exp_sum += exp_xi;
            C[idx] = exp_xi;
        }
        for (int row = 0; row < M; ++row) {
            C[row + col * M] /= exp_sum;
        }
    }
}

int gpu_softmax(const real* __restrict__ A, real* __restrict__ C, int M, int N) {
    dim3 dim_block(BLOCK_SIZE_1D);
    dim3 dim_grid((N + dim_block.x - 1) / dim_block.x);
    gpu_softmax_kernel<<<dim_block, dim_grid>>>(A, C, M, N);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

int gpu_linear(const real* __restrict__ W, const real* __restrict__ x,
                        const real* __restrict__ b, real* __restrict__ z, int in_dim,
                        int out_dim, int batch_size) {
    size_t size_b = out_dim * sizeof(real);
    for (int i = 0; i < batch_size; ++i)
        cudaMemcpy(&z[out_dim * i], b, size_b,
                   cudaMemcpyDeviceToDevice);
    float alpha = 1.0f, beta = 1.0f;
    myGEMM(W, x, z, &alpha, &beta, out_dim, batch_size, in_dim);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

// void gpu_loss_kernel(const real* __restrict__ output,
//                      const real* __restrict__ y, int output_dim, int batch_size) {
    
// }

// void gpu_loss(const real* __restrict__ output, const real* __restrict__ y,
//                 int output_dim, int batch_size) {}

__global__ void gpu_add_kernel(const real*  A,
                               const real*  B, 
                               real*  C,
                               real alpha, real beta,
                               int M,  int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    // printf("%d of %d, %d of %d\n", row, M, col, N);
    if (col < N && row < M) {
        int idx = col * M + row;
        if (idx >= M*N)
            printf("error in idx%d", idx);
        C[idx] = alpha * A[idx] + beta * B[idx];
    }
}

/**
 * C = alpha * A + beta * B
 * A, B: M * N matrix
 * alpha, beta: scalar
 * M, B: Dimension
*/
int gpu_add(const real* A, const real* B,
             real* C, const real& alpha, const real& beta,
             const int& M, const int& N) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_SIZE/BLOCK_DIM_X);
    int grid_x = (N + dim_block.x - 1) / dim_block.x;
    int grid_y = (M + dim_block.y - 1) / dim_block.y;
    dim3 dim_grid(grid_x, grid_y);
    printf("M: %d, N: %d\n", M, N);
    gpu_add_kernel<<<dim_grid, dim_block>>>(A, B, C, alpha, beta, M, N);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

__global__ void gpu_element_multiply_kernel(const real* __restrict__ A,
                                            const real* __restrict__ B,
                                            real* __restrict__ C, real alpha,
                                            int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = col * M + row;
    if (col < N && row < M) {
        C[idx] = A[idx] * B[idx] * alpha;
    }
}

int gpu_element_multiply(const real* __restrict__ A,
                          const real* __restrict__ B, real* __restrict__ C,
                          const real& alpha, const int& M, const int& N) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_SIZE / BLOCK_DIM_X);
    int grid_x = (N + dim_block.x - 1) / dim_block.x;
    int grid_y = (M + dim_block.y - 1) / dim_block.y;
    dim3 dim_grid(grid_x, grid_y);
    gpu_element_multiply_kernel<<<dim_grid, dim_block>>>(A, B, C, alpha, M, N);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

__global__ void transpose_kernel(const real* __restrict__ A,
                                 real* __restrict__ B, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_a = col * M + row;
    int idx_b = row * N + col;
    if (col < N && row < M) {
        B[idx_b] = A[idx_a];
    }
}

int transpose(const real* __restrict__ A, real* __restrict__ B, const int& M, const int & N) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_SIZE / BLOCK_DIM_X);
    int grid_x = (N + dim_block.x - 1) / dim_block.x;
    int grid_y = (M + dim_block.y - 1) / dim_block.y;
    dim3 dim_grid(grid_x, grid_y);
    transpose_kernel<<<dim_grid, dim_block>>>(A, B, M, N);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}

__global__ void sum_row_kernel(const real* __restrict__ A, real* __restrict__ B,
                               int M, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M) {
        real sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row + i * M];
        }  
        B[row] = sum; 
    }
}

int sum_row(const real* __restrict__ A, real* __restrict__ B, const int& M,
             const int& N){
    dim3 dim_block(BLOCK_SIZE_1D);
    dim3 dim_grid((M + dim_block.x - 1) / dim_block.x);
    sum_row_kernel<<<dim_grid, dim_block>>>(A, B, M, N);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

__global__ void matrix_add_const_kernel(const real* __restrict__ src,
                                        real* __restrict__ dst, int c, int M,
                                        int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < N && row < M) {
        int idx = row + col * M;
        dst[idx] = src[idx] + c;
    }
}

int matrix_add_const(const real* src, real* dst, const int& c, const int& M, const int& N) {
    dim3 dim_block(BLOCK_DIM_X, BLOCK_SIZE / BLOCK_DIM_X);
    int grid_x = (N + dim_block.x - 1) / dim_block.x;
    int grid_y = (M + dim_block.y - 1) / dim_block.y;
    dim3 dim_grid(grid_x, grid_y);
    matrix_add_const_kernel<<<dim_grid, dim_block>>>(src, dst, c, M, N);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

__global__ void l2norm_kernel(const real* src, real* col_sum, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    real s = 0.0f;
    if (col < N) {
        for(int row = 0; row < N; ++row) {
            s += src[col * M + row] * src[col * M + row];
        }
        col_sum[col] = s;
    }
}


real l2norm(const real* src, const int& M, const int& N) {
    real l2sum = 0.0f;
    dim3 dim_block(BLOCK_SIZE_1D);
    dim3 dim_grid((N + dim_block.x - 1) / dim_block.x);
    real* d_col_sum;
    real h_col_sum[N];
    cudaMalloc((void**) &d_col_sum, sizeof(real) * N);
    l2norm_kernel<<<dim_grid, dim_block>>>(src, d_col_sum, M, N);
    cudaMemcpy(h_col_sum, d_col_sum, N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        l2sum += h_col_sum[i];
    cudaFree(d_col_sum);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
                __FILE__, __LINE__);
        return -1;
    }
    return sqrt(l2sum);
}