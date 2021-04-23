//
// Created by Zero on 2021/4/23.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "texture.h"
#else
    #error "this file just for cuda shader include"
#endif