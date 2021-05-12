//
// Created by Zero on 2021/4/29.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "bxdf.cpp"
    #include "bsdf.cpp"
#else
    #error "this file just for cuda shader include"
#endif