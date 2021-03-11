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
//    err = cuCtxCreate(&ctx, 0, device);
//    cuCtxSetCurrent(ctx);
//    CUstream stream;
//    cuStreamCreate(&stream, 0);



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
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {0};

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **) &dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **) &dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    int minGridSize;
    int blockSize;
    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0, 0);
    int gridSize = (size + blockSize - 1) / blockSize;
    vector<void *> args = {&dev_c, &dev_a, &dev_b};
    auto r = cuLaunchKernel(func, 1, 1, 1, size, 1,
                            1, 1024, stream, args.data(), nullptr);

    auto css = cudaStreamSynchronize(stream);
    cudaStatus = cudaMemcpy(c, (const uint8_t *)dev_c + sizeof(int), 3 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
//    cout << r;
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    for (int i = 0; i < size; ++i) {
        cout << c[i] << endl;
    }
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
//    buffer_a.upload_async(dispatcher, a, size);
//    buffer_b.upload_async(dispatcher, b, size);


    buffer_a.upload( a, size);
    buffer_b.upload( b, size);
//    buffer_b.download_async(dispatcher,b, size);
    auto stream2 = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
    dispatcher.wait();
    auto pc = buffer_c.ptr<CUdeviceptr>();
    auto pa = buffer_a.ptr<CUdeviceptr>();
    auto pb = buffer_b.ptr<CUdeviceptr>();

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

void test3() {
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));
    CU_CHECK(cuCtxSetCurrent(ctx));
    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, 0));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, ptxCode));
    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, module, "addKernel"));

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {};

    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
    void * dev_c = 0;

    void * p = (void *)dev_a;

    CU_CHECK(cuMemAlloc(&dev_a, sizeof(int) * size));
    CU_CHECK(cuMemAlloc(&dev_b, sizeof(int) * size));
    CU_CHECK(cuMemAlloc((CUdeviceptr*)&dev_c, sizeof(int) * size));

    CU_CHECK(cuMemcpyHtoD(dev_a, a, sizeof(int) * size));
    CU_CHECK(cuMemcpyHtoD(dev_b, b, sizeof(int) * size));
    CU_CHECK(cuMemcpyHtoD((CUdeviceptr)dev_c, c, sizeof(int) * size));

    vector<void *> args = {&dev_c, &dev_a, &dev_b};

    CU_CHECK(cuLaunchKernel(func, 1, 1, 1, size, 1,
                            1, 1024, stream, args.data(), nullptr));

//    cuStreamSynchronize(stream);
    CU_CHECK(cuMemcpyDtoH(c, (CUdeviceptr)dev_c, sizeof(int) * size));

    for (int i : c) {
        cout << i << endl;
    }

    CU_CHECK(cuMemFree(dev_a));
    CU_CHECK(cuMemFree(dev_b));
    CU_CHECK(cuMemFree((CUdeviceptr)dev_c));

}

void test_driver_api() {
    using namespace luminous;
    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("addKernel");

//    auto stream = static_cast<CUDADispatcher>(dispatcher.impl_mut())->stream;

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {};

    auto buffer_a = device->allocate_buffer<int>(size);
    auto buffer_b = device->allocate_buffer<int>(size);
    auto buffer_c = device->allocate_buffer<int>(size);

    buffer_a.upload(a, size);
    buffer_b.upload(b, size);

    auto pa = buffer_a.ptr();
    auto pb = buffer_b.ptr();
    auto pc = buffer_c.ptr();

    kernel->launch(dispatcher, {&pc, &pa, &pb});

    dispatcher.wait();

//    dispatcher.then([&](){
        buffer_c.download(c, size - 2, 1);
        for (int i = 0; i < size; ++i) {
            cout << c[i] << endl;
        }
//    });
}

int main() {
    test_driver_api();
//    test3();
//    test2();
//    test1();

    return 0;
}