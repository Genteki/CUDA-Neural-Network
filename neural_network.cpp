#include "neural_network.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"
#include "mpi.h"
#include "utils/common.h"

#define MPI_SAFE_CALL(call)                                              \
    do {                                                                 \
        int err = call;                                                  \
        if (err != MPI_SUCCESS) {                                        \
            fprintf(stderr, "MPI error %d in file '%s' at line %i", err, \
                    __FILE__, __LINE__);                                 \
            exit(1);                                                     \
        }                                                                \
    } while (0)

real norms(NeuralNetwork& nn) {
    real norm_sum = 0;

    for (int i = 0; i < nn.num_layers; ++i) {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::Mat<real> A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
    real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
    real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
    real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
    real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

    int ow = 15;

    if (iter == 0) {
        error_file << std::left << std::setw(ow) << "Iteration" << std::left
                   << std::setw(ow) << "Max Err W0" << std::left
                   << std::setw(ow) << "Max Err W1" << std::left
                   << std::setw(ow) << "Max Err b0" << std::left
                   << std::setw(ow) << "Max Err b1" << std::left
                   << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
                   << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
                   << std::left << std::setw(ow) << "L2 Err b1"
                   << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left
               << std::setw(ow) << max_errW0 << std::left << std::setw(ow)
               << max_errW1 << std::left << std::setw(ow) << max_errb0
               << std::left << std::setw(ow) << max_errb1 << std::left
               << std::setw(ow) << L2_errW0 << std::left << std::setw(ow)
               << L2_errW1 << std::left << std::setw(ow) << L2_errb0
               << std::left << std::setw(ow) << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::Mat<real> a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::Mat<real> a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::Mat<real> da1 = nn.W[1].t() * diff;

    arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg) {
    int N = yc.n_cols;
    real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    real data_loss = ce_sum / N;
    real reg_loss = 0.5 * reg * norms(nn);
    real loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for (int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads) {
    real h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for (int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for (int j = 0; j < nn.W[i].n_rows; ++j) {
            for (int k = 0; k < nn.W[i].n_cols; ++k) {
                real oldval = nn.W[i](j, k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                real fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                real fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for (int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for (int j = 0; j < nn.b[i].size(); ++j) {
            real oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            real fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            real fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1) / batch_size;

        for (int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
            arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
            arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if (print_every > 0 && iter % print_every == 0) {
                if (grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch "
                          << epoch << "/" << epochs << " = "
                          << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for (int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for (int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is
               zero, it saves for the first batch of each epoch to avoid saving
               too many large files. Note that for the first time, you have to
               run debug and serial modes together. This will run the following
               function and write out files to CPUmats folder. In the later runs
               (with same parameters), you can use just the debug flag to output
               diff b/w CPU and GPU without running CPU version */
            if (print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if (debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

struct NNCache {
    // Weights, bias, outputs
    real *W1, *b1, *z1, *a1; // layer1
    real *W2, *b2, *z2, *prediction; // layer2
    // gradient
    real *dW1, *db1, *da1, *dz1;
    real *dW2, *db2, *dz2;
    // temp
    real *temp;

    NNCache(int input_dim, int output_dim, int o1, int batch_size) {
        cudaMalloc((void**)&W1, input_dim * o1 * sizeof(real));
        cudaMalloc((void**)&b1, o1 * 1 * sizeof(real));
        cudaMalloc((void**)&z1, o1 * batch_size * sizeof(real));
        cudaMalloc((void**)&a1, o1 * batch_size * sizeof(real));
        cudaMalloc((void**)&dW1, input_dim * o1 * sizeof(real));
        cudaMalloc((void**)&db1, o1 * 1 * sizeof(real));
        cudaMalloc((void**)&da1, o1 * batch_size * sizeof(real));
        cudaMalloc((void**)&dz1, o1 * batch_size * sizeof(real));

        cudaMalloc((void **)&W2, o1 * output_dim * sizeof(real));
        cudaMalloc((void **)&b2, output_dim * 1 * sizeof(real));
        cudaMalloc((void**)&z2, output_dim * batch_size * sizeof(real));
        cudaMalloc((void**)&prediction, output_dim * batch_size * sizeof(real));
        cudaMalloc((void**)&dW2, o1 * output_dim * sizeof(real));
        cudaMalloc((void**)&db2, output_dim * 1 * sizeof(real));
        cudaMalloc((void**)&dz2, output_dim * batch_size * sizeof(real));
    }
    ~NNCache() {
        cudaFree(W1); cudaFree(b1); cudaFree(z1); cudaFree(a1);
        cudaFree(W2); cudaFree(b2); cudaFree(z2); cudaFree(prediction);
        cudaFree(dW1); cudaFree(db1); cudaFree(da1); cudaFree(dz1);
        cudaFree(dW2); cudaFree(db2); cudaFree(dz2);
    }
};

void gpu_forward(NeuralNetwork& nn, real *X, NNCache &cache, int batch_size) {
    int input_dim = nn.H[0];
    int o1 =nn.H[1];
    int output_dim = nn.H[2];
    // layer 1: z1 = W1 * X + b1; a1 = sigmoid(z1);
    gpu_linear(cache.W1, X, cache.b1, cache.z1, input_dim, o1, batch_size);
    gpu_sigmoid(cache.z1, cache.a1, o1, batch_size);
    // layer 2: z2 = W2 * a1 + b2; out = softmax(z2);
    gpu_linear(cache.W2, cache.a1, cache.b2, cache.z2, o1, output_dim, batch_size);
    gpu_softmax(cache.z2, cache.prediction, output_dim, batch_size);
}

void gpu_backpropogation(NeuralNetwork& nn, real* __restrict__ X,
                              real* __restrict__ y, NNCache& cache,
                              int batch_size, real reg, real num_process) {
    int input_dim = nn.H[0];
    int o1 = nn.H[1];
    int output_dim = nn.H[2];
    real normalizer = 1.0f / ((real)batch_size * num_process);
    real reg_normalizer = reg / (real)num_process;

    /*     partial derivative of the second layer    */
    // 1. z2: d(C)/d(z2) = prediction - y
    gpu_add(cache.prediction, y, cache.dz2, normalizer, -normalizer, output_dim, batch_size);
    // 2. w2: dC / dW2 = dC/dz2 * (a1).T + reg * W2
    real *temp = nullptr;
    cudaMalloc((void**)&temp, sizeof(real) * o1 * batch_size);
    transpose(cache.a1, temp, o1, batch_size);
    float f1 = 1.0f, f0 = 0.0f;
    myGEMM(cache.dz2, temp, cache.dW2, &f1, &reg_normalizer, output_dim, o1, batch_size);
    cudaFree(temp);
    // 3. b2: dC / db2 = dC/dz2 , but we do average(sum) here
    sum_row(cache.dz2, cache.b2, output_dim, batch_size);
    // 4. a1: dC / da1 = W2.T * dC / dz2
    cudaMalloc((void**)&temp, sizeof(real) * output_dim * o1);
    transpose(cache.W2, temp, output_dim, o1);
    myGEMM(temp, cache.dz2, cache.da1, &f1, &f0, o1, batch_size, output_dim);
    cudaFree(temp);

    /*     partial derivative of the first layer    */
    // 1. z1: dc / dz1 = (dc / da1) o sigmoid(z1) o [1 - sigmoid(z1)]
    cudaMalloc((void**)&temp, sizeof(float) * o1 * batch_size);
    matrix_add_const(cache.a1, temp, -1, o1, batch_size);
    gpu_element_multiply(cache.da1, cache.a1, cache.dz1, 1.0f, o1, batch_size);
    gpu_element_multiply(cache.dz1, temp, cache.dz1, -1.0f, o1, batch_size);
    cudaFree(temp);
    // 2. W1
    cudaMalloc((void**)&temp, sizeof(real) * input_dim * batch_size);
    transpose(X, temp, input_dim, batch_size);
    myGEMM(cache.dz1, temp, cache.dW1, &f1, &reg_normalizer, o1,
           input_dim, batch_size);
    cudaFree(temp);
    // 3. b1
    sum_row(cache.dz1, cache.b1, o1, batch_size);
}

void zero_gradient(NNCache& cache) {
    
}

/*
 * TODO
 * Train the neural network nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check,
                    int print_every, int debug) {
    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0) ? X.n_cols : 0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* * 
     * We could load all data in GPU memory at once, or load single batch one by one
     * It depends on the size of dataset
     * In our situation, we could directly load them at once. 
     * */
    int input_dim = nn.H[0];
    int o1 = nn.H[1];
    int output_dim = nn.H[2];
    // real *X_batch, *y_batch;
    // cudaMalloc((void**)X_batch, sizeof(real) * input_dim * batch_size);
    // cudaMalloc((void**)y_batch, sizeof(real) * output_dim * batch_size);
    int num_batches = (N + batch_size - 1) / batch_size;
    real *d_X, *d_y;
    int samples_per_proc = ceil((float)N / (float)num_procs);
    int col_start = rank * samples_per_proc;
    int col_end = (rank == (num_procs-1)) ?  N : (rank + 1) * samples_per_proc;
    int samples_this_proc = col_end - col_start;
    cudaMalloc((void**)&d_X, (samples_this_proc * input_dim) * sizeof(real));
    cudaMalloc((void**)&d_y, (samples_this_proc * output_dim) * sizeof(real));
    cudaMemcpy(d_X, X.colptr(col_start), (samples_this_proc * input_dim) * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.colptr(col_start), (samples_this_proc * output_dim) * sizeof(real), cudaMemcpyHostToDevice);

    NNCache cache(input_dim, output_dim, o1, batch_size);
    real* h_W1 = new real[o1 * input_dim]();
    real* h_b1 = new real[o1 * 1]();
    real* h_W2 = new real[output_dim * o1]();
    real* h_b2 = new real[output_dim * 1]();
    real* h_dW1 = new real[o1 * input_dim]();
    real* h_db1 = new real[o1 * 1]();
    real* h_dW2 = new real[output_dim * o1]();
    real* h_db2 = new real[output_dim * 1]();

    /* iter is a variable used to manage debugging. It increments in the inner
       loop and therefore goes from 0 to epochs*num_batches */
    int iter = 0;
    printf("start training\n");

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // int num_batches = (N + batch_size - 1) / batch_size;
        for (int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each
             * MPI node
             * 2. compute each sub-batch of images' contribution to network
             * coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with
             * `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */

            int batch_col_start = batch * batch_size;
            int batch_col_end = std::min(col_start + batch_size, samples_this_proc);
            int this_batch_size = batch_col_end - batch_col_start;
            real *d_X_batch = d_X + batch_col_start * input_dim;
            real *d_y_batch = d_y + batch_col_start * output_dim;
            // 1. copy cache to device
            cudaMemcpy(cache.W1, h_W1, sizeof(real) * input_dim * o1, cudaMemcpyHostToDevice);
            cudaMemcpy(cache.W2, h_W2, sizeof(real) * o1 * output_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(cache.b1, h_b1, sizeof(real) * o1 * 1, cudaMemcpyHostToDevice);
            cudaMemcpy(cache.b2, h_b2, sizeof(real) * output_dim * 1, cudaMemcpyHostToDevice);
            // 2. forward
            gpu_forward(nn, d_X_batch, cache, this_batch_size);
            // 3. zero gradient
            // TODO
            // 4. backpropogation
            gpu_backpropogation(nn, d_X_batch, d_y_batch, cache,
                                this_batch_size, reg, num_procs);
            // copy derivatives back to host
            cudaMemcpy(h_dW1, cache.dW1, sizeof(real) * input_dim * o1, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dW2, cache.dW2, sizeof(real) * output_dim * o1, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_db1, cache.db1, sizeof(real) * o1 * 1, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_db2, cache.db2, sizeof(real) * output_dim * 1, cudaMemcpyDeviceToHost);
            // 5. gradient desent
            arma::Mat<real> dW1(size(nn.W[0]), arma::fill::zeros);
            arma::Mat<real> dW2(size(nn.W[1]), arma::fill::zeros);
            arma::Mat<real> db1(size(nn.b[0]), arma::fill::zeros);
            arma::Mat<real> db2(size(nn.b[1]), arma::fill::zeros);
            MPI_SAFE_CALL(MPI_Allreduce(h_dW1, dW1.memptr(), (input_dim * o1),
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(h_dW2, dW2.memptr(), (o1 * output_dim),
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(h_db1, db1.memptr(), o1, MPI_FP,
                                        MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(h_db2, db2.memptr(), output_dim, MPI_FP,
                                        MPI_SUM, MPI_COMM_WORLD));
            std::cout << "Device dW:" << l2norm(cache.dW1, o1, input_dim)
                      << l2norm(cache.dW2, o1, output_dim) << std::endl;
            std::cout << "Host dW:" << arma::norm(dW2, 2) << arma::norm(db2, 2) << std::endl;
            nn.W[0] -= learning_rate * dW1;
            nn.W[1] -= learning_rate * dW2;
            nn.b[0] -= learning_rate * db1;
            nn.b[1] -= learning_rate * db2;
            return;
            // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=
            // //
            //                    POST-PROCESS OPTIONS //
            // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=
            // //
            if (print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if (debug && rank == 0 && print_flag) {
                // TODO
                // Copy data back to the CPU

                /* The following debug routine assumes that you have already
                 updated the arma matrices in the NeuralNetwork nn.  */
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    // TODO
    // Copy data back to the CPU

    error_file.close();

    // Free memory
    cudaFree(d_X); cudaFree(d_y);
    delete[] h_W1;
    delete[] h_b1;
    delete[] h_W2;
    delete[] h_b2;
    delete[] h_dW1;
    delete[] h_db1;
    delete[] h_dW2;
    delete[] h_db2;
}
