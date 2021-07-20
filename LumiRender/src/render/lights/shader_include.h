//
// Created by Zero on 2021/4/7.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "light.cpp"
    #include "area_light.cpp"
    #include "point_light.cpp"
    #include "spot_light.cpp"
#else
    #error "this file just for cuda shader include"
#endif