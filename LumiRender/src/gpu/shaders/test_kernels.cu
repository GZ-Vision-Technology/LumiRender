//
// Created by Zero on 2021/2/14.
//



#include "cuda_util.cuh"
//#include "render/samplers/shader_include.h"

extern "C" __global__ void addKernel(unsigned long long _, int d,int *a, int *b, long long c)
{
//    using namespace luminous;

    int i = 1;
    long long l = 132456;

    *b = l + i;

//    PCGSampler pcg_sampler(100);
//    float f = 0;
//
//    Sampler sampler(pcg_sampler);
//    if (d == 1) {
//        for(int i = 0; i < 1000000; ++i) {
//            f += sampler.next_1d();
//        }
//    } else {
//        for(int i = 0; i < 1000000; ++i) {
//            f += pcg_sampler.next_1d();
//        }
//    }
//
//
//    *b = int(f);
////    printf("%d\n", d);
}