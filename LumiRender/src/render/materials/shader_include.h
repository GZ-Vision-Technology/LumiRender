//
// Created by Zero on 2021/5/1.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "material.h"
    #include "matte.cpp"
    #include "ai_material.cpp"
#else
    #error "this file just for cuda shader include"
#endif