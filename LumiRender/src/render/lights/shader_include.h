//
// Created by Zero on 2021/4/7.
//


#pragma once

#if defined(__CUDACC__)
    #include "light.cpp"
    #include "area_light.cpp"
    #include "point_light.cpp"
    #include "spot_light.cpp"
    #include "envmap.cpp"
#else
    #include "light.h"
    #include "area_light.h"
    #include "point_light.h"
    #include "spot_light.h"
    #include "envmap.h"
#endif