//
// Created by Zero on 2021/2/20.
//

#include "iostream"
#include "util/image.h"
#include "cpu/texture/mipmap.h"

using namespace luminous;
using namespace std;



int main() {

    auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\HelloWorld.png)";
    auto path2 = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\test.hdr)";

    auto image = Image::load(path, LINEAR);
    image.save(path2);

//    auto mipmap = MIPMap(image);


    return 0;
}