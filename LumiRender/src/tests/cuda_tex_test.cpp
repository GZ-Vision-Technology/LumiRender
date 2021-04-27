//
// Created by Zero on 2021/4/24.
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
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"

#include "render/samplers/sampler.h"
#include "util/image.h"

using namespace std;

extern "C" char ptxCode[];
using namespace luminous;
void test_tex_load(luminous::uchar uc) {
    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();
    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_tex_sample");
    auto pixel = new luminous::uchar4[1];
    *pixel = make_uchar4(uc);
    printf("%d ", uint32_t(uc));
    auto image2 = Image(luminous::utility::PixelFormat::RGBA8U, (byte*)pixel, make_uint2(1));

    auto texture = device->allocate_texture(image2.pixel_format(), image2.resolution());
    texture.copy_from(image2);

    auto handle = texture.tex_handle<CUtexObject>();
    float u = 0;
    float v = 0;
    kernel->configure(make_uint3(1),make_uint3(1));
    kernel->launch(dispatcher, {&handle,&u,&v});
    dispatcher.wait();
}

int main() {
    for (int i = 0; i < 256; ++i) {
        test_tex_load(i);
    }
    
    return 0;
}