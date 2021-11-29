//
// Created by Zero on 2021/5/1.
//


#pragma once

#if defined(__CUDACC__)
    #include "material.cpp"
    #include "matte.cpp"
    #include "dieletric.cpp"
    #include "conductor.cpp"
    #include "ai_material.cpp"
#else
    #include "material.h"
    #include "matte.h"
    #include "dieletric.h"
    #include "conductor.h"
    #include "ai_material.h"
#endif