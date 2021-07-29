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

#include "render/samplers/sampler.h"
#include "render/distribution/distribution_handle.h"
#include "render/distribution/envmap_distribution.h"

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
    config.set_full_type("LCGSampler");
    config.spp = 9;
    auto sampler = Sampler::create(config);
    auto buffer = device->allocate_buffer<Sampler>(1);
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
    config.set_full_type("PCGSampler");
    config.spp = 9;
    auto sampler = Sampler::create(config);
    auto buffer = device->allocate_buffer<Sampler>(1);
    buffer.upload(&sampler);
    auto ps = buffer.ptr();

    Managed<Sampler> s;

    vector<Sampler> v;

    v.push_back(sampler);
    v.push_back(sampler);

//    s.reset(&sampler, device);
    s.reset(v,device);
    cout << v[0].to_string() << endl;
    s.synchronize_to_gpu();
    ps = s.device_ptr();
    kernel->launch(dispatcher, {&ps});
    dispatcher.wait();
    s.synchronize_to_cpu();
    cout << s[0].next_1d() << endl;

    cout << sizeof(vector<Sampler>);

}
using namespace luminous;
void test_distribution() {
    vector<float> vec;
    int nu = 5;
    int nv = 5;
    for (int i = 0; i < nu * nv; ++i) {
        vec.push_back(i);
    }

    auto ed = EnvmapDistribution();
    ed.init(vec, nu, nv);
    ed.init_on_host();
    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();

    ed.init_on_device(device);
//    ed.synchronize_to_gpu();
}

int main() {

//    test_managed();

    test_distribution();

//    test_driver_api();
//    test_kernel_sampler();
//    test3();
//    test2();
//    test1();

    return 0;
}