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
    auto path2 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr.hdr)";
    auto path3 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr222.png)";

    auto image = Image::load(path2, LINEAR);

    auto texture = device->allocate_texture(image.pixel_format(), image.resolution());
    texture.copy_from(image);

    auto image2 = texture.download();

    image2.save_image(path3);

}

int main() {
    test_tex_load();
    return 0;
}