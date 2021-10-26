//
// Created by Zero on 08/09/2021.
//


#pragma once

#if defined(__CUDACC__)
    #include "scene_data.cpp"
#else
    #include "scene_data.h"
#endif