//
// Created by Zero on 2021/2/14.
//

#include "render/scattering/microfacet.cpp"

extern "C" __global__ void addKernel(unsigned long long _, int d,int *a, int *b, long long c)
{
    using namespace luminous;
    Microfacet microfacet(0.5,0.5);
    luminous::float3 wh = luminous::make_float3(1.f,1.f,1.f);
    *a = microfacet.D(wh);
}