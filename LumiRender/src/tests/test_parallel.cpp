//
// Created by Zero on 06/09/2021.
//

#include "util/parallel.h"
#include <iostream>
#include <functional>
#include "cpu/cpu_impl.h"
#include "cpu/accel/embree_util.h"
#include "windows.h"
#include "util/thread_pool.h"

using namespace std;
using namespace luminous;

void fun() {
    parallel_for(3, [&](uint, uint tid) {
        printf("asdf %d\n", int(0));
    });
}

int main() {
    setbuf(stdout, NULL);
//    Ray ray[16];
////    printf("now tid is %d \n", GetCurrentThreadId());
//    parallel_for(1, [&](uint, uint tid){
//        printf("tid out :%d\n", (int)tid);
//        fun();
//        printf("tid out :%d\n", (int)tid);
//    });

    init_thread_pool(9);

    parallelFor(20, [&](uint idx, uint tid) {
        printf("%u %u\n", idx, tid);
    },1);

//    std::function < void(void**, uint32_t) > func = [&](void**, uint idx) {
//        printf("%u \n", idx);
//    };
//
//    int count = 4;
//
//    auto device = create_cpu_device();
//
//    auto kernel = create_cpu_kernel(func);
//
//    auto dispatcher = device->new_dispatcher();
//
//    kernel->launch(count, dispatcher, func);
//
//    dispatcher.wait();
    getchar();
    return 0;
}