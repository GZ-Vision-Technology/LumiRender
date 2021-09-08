//
// Created by Zero on 06/09/2021.
//

#include "util/parallel.h"
#include <iostream>
#include <functional>
#include "cpu/cpu_impl.h"
#include "cpu/accel/embree_util.h"
using namespace std;
using namespace luminous;


int main() {
    Ray ray[16];

//    parallel_for(1, [&](uint, uint){
//auto r = luminous::to_RTCRayN<4>(ray);
//auto r = luminous::to_RTCRayN<8>(ray);
auto r = luminous::to_RTCRayN<16>(ray);
//    });




    std::function < void(void**, uint32_t) > func = [&](void**, uint idx) {
        printf("%u \n", idx);
    };

    int count = 4;

    auto device = create_cpu_device();

    auto kernel = create_cpu_kernel(func);

    auto dispatcher = device->new_dispatcher();

    kernel->launch(count, dispatcher, func);

    dispatcher.wait();

    return 0;
}