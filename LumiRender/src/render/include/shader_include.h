//
// Created by Zero on 2021/5/4.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
//    #include "shader_data.cpp"
#else
    #error "this file just for cuda shader include"
#endif