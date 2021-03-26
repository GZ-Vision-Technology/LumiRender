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
#include "core/backend/managed.h"

#include "render/samplers/sampler_handle.h"

using namespace std;

extern "C" char ptxCode[];

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

    buffer_c.download(c, size - 2, 1);
    for (int i = 0; i < size; ++i) {
        cout << c[i] << endl;
    }
}

void test_kernel_sampler() {
    using namespace luminous;

    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_sampler2");

    auto config = SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
    auto sampler = SamplerHandle::create(config);
    auto buffer = device->allocate_buffer<SamplerHandle>(1);
    buffer.upload(&sampler);
    auto ps = buffer.ptr();

    kernel->launch(dispatcher, {&ps});
}

void test_managed() {
    using namespace luminous;

    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_sampler2");

    auto config = SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
    auto sampler = SamplerHandle::create(config);
    auto buffer = device->allocate_buffer<SamplerHandle>(1);
    buffer.upload(&sampler);
    auto ps = buffer.ptr();

    int a[10] = {1,2,3};



    auto hd = Managed_old(sampler, device);
    hd.synchronize_to_gpu();

    Managed_old<SamplerHandle> s;
    s.reset(sampler, device);
    s.synchronize_to_gpu();
    cout << s->to_string();
    ps = s.device_ptr();
    kernel->launch(dispatcher, {&ps});
    dispatcher.wait();
    s.synchronize_to_cpu();
    cout << s->next_1d() << endl;
}

int main() {

    test_managed();
//    test_driver_api();
//    test_kernel_sampler();
//    test3();
//    test2();
//    test1();

    return 0;
}