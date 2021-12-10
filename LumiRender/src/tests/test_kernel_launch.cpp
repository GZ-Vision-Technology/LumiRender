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

#include "render/include/kernel.h"
#include "render/samplers/sampler.h"
#include "render/lights/light.h"
#include "base_libs/sampling/distribution.h"
#include "test_light.h"
#include "util/clock.h"

using namespace std;

extern "C" char ptxCode[];

void add_func(int n, int a, int *, int*,int*) {
    printf("adfdfad");
}

void test_driver_api() {
    using namespace luminous;
    auto device = create_cuda_device();
    Dispatcher dispatcher = device->new_dispatcher();

    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel_handle = cudaModule->get_kernel_handle("addKernel");
//    cout << ptxCode << endl;
    Kernel<decltype(&add_func)> kernel(add_func);
    kernel.set_cu_function(kernel_handle);

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {};

    auto buffer_a = device->create_buffer<int>(size);
    auto buffer_b = device->create_buffer<int>(size);
    auto buffer_c = device->create_buffer<int>(size);

    buffer_a.upload(a, size);
    buffer_b.upload(b, size);

    int* pa = buffer_a.ptr<int*>();
    int* pb = buffer_b.ptr<int*>();
    int* pc = buffer_c.ptr<int*>();
    int n = 5;
    Clock clock1;

    kernel.launch(dispatcher, 1, pa, pb, pc);

//    int nitem = 5;
////    kernel->launch(dispatcher, nitem,pc, pa, pb);
//    kernel->launch(dispatcher, pc, pa, pb);
//
    dispatcher.wait();
//    cout << clock1.elapse_ms() << endl;

}

void test_kernel_sampler() {
    using namespace luminous;

    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();


    auto config = SamplerConfig();
    config.set_full_type("LCGSampler");
    config.spp = 9;
    auto sampler = Sampler::create(config);
    int count = 50000000;
    float f = 0;
//    sampler.start_pixel_sample(make_uint2(0), 0, 0);
//    for (int i = 0; i < count; ++i) {
//        f = sampler.next_1d();
////        cout << f << endl;
//    }
    sampler.start_pixel_sample(make_uint2(0), 0, 0);

    LCGSampler lcg_sampler = *sampler.get<LCGSampler>();

    uint32_t state = 1576399551;

    int cc;
    cin >> cc;
    if (cc > 1) {
        state = 0;
    }
    Clock clock;
    for (int i = 0; i < count; ++i) {
//        auto nn = lcg_func(state);
        auto nn = rand();
        f += nn;

//        cout << f << endl;
    }

//    cout << clock.elapse_ms() << endl;
    
//    buffer.upload(&sampler);
//    auto ps = buffer.ptr();
//
//    kernel->launch(dispatcher, ps);
}

int main() {

//    test_managed();

//    test_driver_api();

//    test_memory();

    test_kernel_sampler();
//    test3();
//    test2();
//    test1();

    return 0;
}