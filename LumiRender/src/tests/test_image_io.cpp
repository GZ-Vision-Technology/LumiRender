//
// Created by Zero on 2021/2/20.
//

#include "iostream"
#include "util/image.h"
using namespace luminous;
using namespace std;



int main() {

    auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\HelloWorld.png)";
    auto path2 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr.hdr)";

    auto image = Image::load(path, LINEAR);
    image.save_image(path2);

//    auto [rgb, res] = load_image(path);
//
//    save_image(path2, rgb.get(), res);

    return 0;
}