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
#include "gpu/framework/cuda_device.h"
#include "gpu/framework/cuda_kernel.h"
#include "gpu/framework/cuda_module.h"

using namespace std;

extern "C" char ptxCode[];

void test1() {
    CUresult err = cuInit(0);
    CUdevice device;
    err = cuDeviceGet(&device, 0);
    CUcontext ctx;
    err = cuCtxCreate(&ctx, 0, device);


    string s = ptxCode;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CUmodule module;

//    CUstream stream;
//    cuStreamCreate(&stream, 0);
    auto a0 = cuModuleLoadData(&module, ptxCode);

    CUfunction func;


    CUfunction func2;
    auto a1 = cuModuleGetFunction(&func, module, "addKernel");
    auto a3 = cuModuleGetFunction(&func2, module, "testKernel");

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {0};

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
    cudaStatus = cudaMalloc((void **) &dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_b, size * sizeof(int));
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
    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0, 0);
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
}

void test2() {
    using namespace luminous;
    auto device = create_cuda_device();

    const auto &d2 = device;

    string s = ptxCode;


    auto cuda_module = create_cuda_module(ptxCode);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto cuda_kernel = cuda_module->get_kernel("addKernel");

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {};

    auto buffer_a = device->allocate_buffer<int>(size);
    auto buffer_b = device->allocate_buffer<int>(size);
    auto buffer_c = device->allocate_buffer<int>(size);
    auto dispatcher = device->new_dispatcher();
    buffer_a.upload_async(dispatcher, a, size);
//    buffer_b.download_async(dispatcher,b, size);
    auto stream2 = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
    dispatcher.wait();
    auto pc = buffer_c.ptr();
    auto pa = buffer_a.ptr();
    auto pb = buffer_b.ptr();

    vector<void *> args = {&pc, &pa, &pb};

//    cuda_kernel->configure(make_uint3(5), make_uint3(5))
//                .launch(dispatcher,move(args));
//    auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
    CUmodule module;
    auto a0 = cuModuleLoadData(&module, ptxCode);
    CUfunction func;
    auto a1 = cuModuleGetFunction(&func, module, "addKernel");
    auto r = cuLaunchKernel(((CUDAKernel *)(cuda_kernel->impl.get()))->_func, 1, 1, 1, size, 1,
                            1, 1024, stream2, args.data(), nullptr);

    auto css = cudaStreamSynchronize(stream);
//    for (int i = 0; i < size; ++i) {
//        cout << b[i] << "---"<< endl;
//    };
}

int main() {

//    test2();
    test1();

    return 0;
}