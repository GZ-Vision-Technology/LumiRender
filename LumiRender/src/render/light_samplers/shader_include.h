//
// Created by Zero on 2021/4/12.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "uniform.cpp"
    #include "light_sampler.cpp"
#else
    #error "this file just for cuda shader include"
#endif
