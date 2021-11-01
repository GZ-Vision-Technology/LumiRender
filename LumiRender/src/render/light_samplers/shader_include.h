//
// Created by Zero on 2021/4/12.
//


#pragma once

#if defined(__CUDACC__)
    #include "uniform.cpp"
    #include "light_sampler.cpp"
    #include "light_sampler_base.cpp"
#else
    #include "light_sampler.h"
    #include "light_sampler_base.h"
#endif
