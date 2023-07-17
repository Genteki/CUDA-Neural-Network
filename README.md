# Cuda-Neural-Network

It's a 2-layer neural network using CUDA C++, identifying digits from hand-written images.

## Dependencies

* cuda
* [cuda-samples](https://github.com/nvidia/cuda-samples)
* [openmpi](https://www.open-mpi.org)

## Before compilation

- [ ] In `Makefile`, replace the `-arch=compute_80 -code=sm_80` with the version [matching to GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) in `CUDFLAGS`.
- [ ] In `Makefile`, replace the `CUDASAMPLES` with your cuda-samples include path.
- [ ] Make a directory named `data/`, download [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in the directory.

## Compile

``` bash
make
```

## Run

``` bash
./main -g 1
```

## Credits

* `starter-code` from [CME213 Final Project](https://ericdarve.github.io/cme213-spring-2021/) by Eric Darve.
* Dataset [MNIST](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
