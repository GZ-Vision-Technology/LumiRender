//
// Created by Zero on 15/10/2021.
//

#include <iostream>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>
#include "base_libs/common.h"
#include "base_libs/lstd/lstd.h"
#include "render/include/kernel.h"


using namespace luminous;

int foo(int x, int y, float a) {
    std::cout << x << " " << y << "  " << a << std::endl;
    return x;
}


using namespace std;

struct A {

};

int main() {
    int x = 4;
    const float y = 6.5f;

    auto device = create_cpu_device();

    auto dispatcher = device->new_dispatcher();

    luminous::Kernel kernel{foo};

    kernel.launch(dispatcher,  y, y);

//    cout << bit_cast<float>(4) << endl;



//    cout << a  << endl;


}