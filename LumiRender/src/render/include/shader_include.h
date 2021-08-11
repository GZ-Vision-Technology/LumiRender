//
// Created by Zero on 2021/5/5.
//

#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "scene_data.cpp"
    #include "interaction.cpp"
#else
    #error "this file just for cuda shader include"
#endif
