//
// Created by Zero on 2021/5/9.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "pt.cpp"
#else
    #error "this file just for cuda shader include"
#endif