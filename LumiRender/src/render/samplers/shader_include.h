//
// Created by Zero on 2021/3/22.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "sampler.cpp"
    #include "independent.cpp"
#else
    #include "sampler.h"
    #include "independent.h"
#endif