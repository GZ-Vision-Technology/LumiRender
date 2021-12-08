//
// Created by Zero on 2021/2/14.
//



#include "cuda_util.cuh"
#include "render/samplers/shader_include.h"

__device__ float lcg(uint32_t &state) {
    constexpr auto lcg_a = 1664525u;
    constexpr auto lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return static_cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
};

__device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

extern "C" __global__ void addKernel(unsigned long long _, int d,int *a, int *b, long long c)
{
    using namespace luminous;

    LCGSampler pcg_sampler(100);
    float f = 0;

    uint32_t state = 0u;

    Sampler sampler(pcg_sampler);
    if (d == 1) {
        for(int i = 0; i < 1000000; ++i) {
            f += sampler.next_1d();
        }
//        printf("sampler\n");
    } else if (d == 2) {
        for(int i = 0; i < 1000000; ++i) {
            f += lcg(state);
        }
//        printf("lcg\n");
    } else {
        for(int i = 0; i < 1000000; ++i) {
            f += pcg_sampler.next_1d();
        }
//        printf("pcg\n");
    }




    *b = int(f);
////    printf("%d\n", d);
}