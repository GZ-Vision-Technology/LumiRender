//
// Created by Zero on 2020/12/29.
//

#include <iostream>
#include <driver_types.h>
#include <string>
#include <vector>
#include <driver_types.h>
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"

#include "gpu/framework/jitify/jitify.hpp"

#include "render/samplers/sampler.h"
#include "render/lights/light.h"
#include "base_libs/sampling/distribution.h"

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

    auto buffer_a = device->create_buffer<int>(size);
    auto buffer_b = device->create_buffer<int>(size);
    auto buffer_c = device->create_buffer<int>(size);

    buffer_a.upload(a, size);
    buffer_b.upload(b, size);

    auto pa = buffer_a.ptr();
    auto pb = buffer_b.ptr();
    auto pc = buffer_c.ptr();
    vector<void *> args{&pc, &pa, &pb};
    int nitem = 5;
//    kernel->launch(dispatcher, nitem,pc, pa, pb);
    kernel->launch(dispatcher, pc, pa, pb);

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
    config.set_full_type("LCGSampler");
    config.spp = 9;
    auto sampler = Sampler::create(config);
    auto buffer = device->create_buffer<Sampler>(1);
    buffer.upload(&sampler);
    auto ps = buffer.ptr();

    kernel->launch(dispatcher, ps);
}

void test_managed() {
    using namespace luminous;

    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_sampler2");

    auto b = device->obtain_buffer(6);

    cout << "Adfdasf" << endl;

}

void test_memory() {
    using namespace luminous;
    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_light");

    LightConfig config;
    config.set_full_type("AreaLight");
    config.instance_idx = 0;
    auto light = Light::create(config);
    light.print();
    printf("%f\n", light.get<AreaLight>()->padded);
    Managed<Light> ml{device.get()};
    ml.push_back(light);
    ml.allocate_device(1);
    ml.synchronize_to_device();
    const Light* ptr = ml.device_data();
    kernel->launch( dispatcher, ptr);
    dispatcher.wait();
}

int main() {

//    test_managed();

//    test_driver_api();

    test_memory();
//    test_kernel_sampler();
//    test3();
//    test2();
//    test1();

    return 0;
}