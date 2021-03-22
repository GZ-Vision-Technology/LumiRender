//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "render/include/interaction.h"
#include "render/films/shader_include.h"
#include "render/samplers/shader_include.h"
#include "render/sensors/shader_include.h"
#include "gpu/framework/optix_params.h"

static __forceinline__ __device__ void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}