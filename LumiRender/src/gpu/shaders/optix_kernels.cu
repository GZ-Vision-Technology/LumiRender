//
// Created by Zero on 2021/2/14.
//

#define GLOBAL extern "C" __global__ void

#include "optix_kernels.h"
#include "gpu/framework/optix_params.h"

extern "C" {
__constant__ luminous::LaunchParams params;
}

GLOBAL __raygen__rg() {

}

GLOBAL __miss__radiance() {

}

GLOBAL __miss__shadow() {

}

GLOBAL __closesthit__radiance() {

}

GLOBAL __closesthit__occlusion() {

}
