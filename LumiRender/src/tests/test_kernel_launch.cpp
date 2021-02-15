//
// Created by Zero on 2020/12/29.
//

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <string>
#include <vector>
#include <driver_types.h>

using namespace std;

extern "C" char ptxCode[];

int main() {


    string s = ptxCode;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CUmodule module;
    auto a0 = cuModuleLoadData(&module, ptxCode);
    CUfunction func;
    CUfunction func2;
    auto a1 = cuModuleGetFunction(&func, module, "addKernel");
    auto a3 = cuModuleGetFunction(&func2, module, "testKernel");

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5 };
    const int b[size] = {10, 20, 30, 40, 50 };
    int c[size] = {0 };

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
    }

    int minGridSize;
    int blockSize;
    cuOccupancyMaxPotentialBlockSize (&minGridSize, &blockSize, func, 0,0, 0);
    int gridSize = (size + blockSize - 1) / blockSize;
    vector<void *> args = {&dev_c, &dev_a, &dev_b};
    auto r = cuLaunchKernel(func, 1, 1, 1, size, 1,
                   1, 1024, stream, args.data(), nullptr);

    auto css = cudaStreamSynchronize(stream);
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
//    cout << r;
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
//    cout << s;
    return 0;
}