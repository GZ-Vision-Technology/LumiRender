//
// Created by Zero on 2021/1/10.
//

#include "cuda_acceleration.h"
#include "jitify/jitify.hpp"
void func() {
    int *p;
    cudaMalloc((void**)&p, 9);
}