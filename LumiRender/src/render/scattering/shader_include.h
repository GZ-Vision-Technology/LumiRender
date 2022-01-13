//
// Created by Zero on 2021/4/29.
//


#pragma once

#if defined(__CUDACC__)
    #include "bsdf_wrapper.cpp"
    #include "diffuse_scatter.cpp"
    #include "microfacet_scatter.cpp"
    #include "specular_scatter.cpp"
    #include "disney_bsdf.cpp"
    #include "bsdfs.cpp"
#else
    #include "microfacet_scatter.h"
    #include "specular_scatter.h"
#endif