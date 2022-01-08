//
// Created by Zero on 2021/2/20.
//

#include "iostream"
#include "util/image.h"
#include "cpu/texture/mipmap.h"

using namespace luminous;
using namespace std;



int main() {

    auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr.exr)";
    auto path2 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr1.hdr)";

    auto image = Image::load(path, LINEAR);

    auto mipmap = MIPMap(image);

//    auto [rgb, res] = load_image(path);
//
//    save_image(path2, rgb.get(), res);

    return 0;
}