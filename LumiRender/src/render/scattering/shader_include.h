//
// Created by Zero on 2021/4/29.
//


#pragma once

#if defined(__CUDACC__)
    #include "bxdf.cpp"
    #include "bsdf_wrapper.cpp"
    #include "dielectric.cpp"
#else
    #include "microfacet_scatter.h"
    #include "specular_scatter.h"
#endif