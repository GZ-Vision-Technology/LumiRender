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

#include "render/samplers/sampler_handle.h"

using namespace std;

extern "C" char ptxCode[];

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
    vector<void *> args{&pc, &pa, &pb};
//    kernel->launch(dispatcher, 5,{&pc, &pa, &pb});
    kernel->launch(dispatcher,args);

    dispatcher.wait();

//    dispatcher.then([&](){
        buffer_c.download(c, size - 2, 1);
        for (int i = 0; i < size; ++i) {
            cout << c[i] << endl;
        }
//    });
}

void test_kernel_sampler() {
    using namespace luminous;
    cout << "adsfdsafsa" << endl;

    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_sampler");

    auto config = SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
//    auto sampler = SamplerHandle::create(config).get<LCGSampler>();
    auto sampler = SamplerHandle::create(config);

//    sampler->next_2d();

    kernel->launch(dispatcher, {&sampler});
}

int main() {
//    test_driver_api();
    test_kernel_sampler();
//    test3();
//    test2();
//    test1();

    return 0;
}