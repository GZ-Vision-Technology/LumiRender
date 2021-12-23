//
// Created by Zero on 2021/2/14.
//

#include "render/scattering/bsdf_data.h"


extern "C" __global__ void addKernel(unsigned long long _, int d,int *a, int *b, long long c ,luminous::BSDFData data)
{
    using namespace luminous;
//    *b = data.color.x;
    *b = data.diffuse_data.color.x;
}