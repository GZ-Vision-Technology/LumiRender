//
// Created by Zero on 24/11/2021.
//


#pragma once

#if defined(__CUDACC__)
    #include "filter_sampler.cpp"
    #include "gaussian.cpp"
    #include "sinc.cpp"
    #include "mitchell.cpp"
    #include "filter.h"
#else
    #include "filter.h"
#endif