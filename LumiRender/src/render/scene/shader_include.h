//
// Created by Zero on 08/09/2021.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "scene_data.cpp"
#else
    #error "this file just for cuda shader include"
#endif