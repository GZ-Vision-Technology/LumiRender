//
// Created by Zero on 2021/3/22.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "film_handle.cpp"
    #include "rgb.cpp"
    #include "g_buffer.cpp"
#else
    #error "this file just for cuda shader include"
#endif