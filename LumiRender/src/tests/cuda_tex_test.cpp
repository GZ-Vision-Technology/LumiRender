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

void test_tex_load() {
    auto device = create_cuda_device();
    auto dispatcher = device->new_dispatcher();
    auto cudaModule = create_cuda_module(ptxCode);
    auto kernel = cudaModule->get_kernel("test_tex_sample");

    for (int i = 0; i < 255; ++i) {
        auto pixel = new luminous::uchar[2];
        uchar uc = i;
        pixel[0] = uc;
        pixel[1] = 256 - uc;
        printf("origin val :%d, con f %f", uint32_t(uc));
        auto image2 = Image(luminous::utility::PixelFormat::R8U, (byte *) pixel, luminous::make_uint2(2u,1u));

        auto texture = device->allocate_texture(image2.pixel_format(), image2.resolution());
        texture.copy_from(image2);

        auto img = texture.download();
        cout << " download val0:" << uint(img.pixel_ptr<decltype(uc)>()[0]) << "  ";
        cout << " download val1:" << uint(img.pixel_ptr<decltype(uc)>()[1]) << "  ";
        auto handle = texture.tex_handle<CUtexObject>();
        float u = 0.499;
        float v = 0;
        kernel->configure(make_uint3(1), make_uint3(1));
        kernel->launch(dispatcher, {&handle, &u, &v});
        dispatcher.wait();
    }
}

int main() {
    test_tex_load();

    return 0;
}