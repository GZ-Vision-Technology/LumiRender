//
// Created by Zero on 2021/4/29.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "bxdf.h"
    #include "bsdf.h"
#else
    #error "this file just for cuda shader include"
#endif