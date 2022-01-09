//
// Created by Zero on 2021/2/20.
//

#include "iostream"
#include "util/image.h"
#include "cpu/texture/mipmap.h"

using namespace luminous;
using namespace std;



int main() {

    auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\HelloWorldq.png)";
    auto path2 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\png2exr.hdr)";

    auto image = Image::load(path2, LINEAR);
//    image.save(path);

    auto mipmap = MIPMap(image);


    return 0;
}