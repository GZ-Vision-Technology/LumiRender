//
// Created by Zero on 2021/5/1.
//


#pragma once

#if defined(__CUDACC__)
    #include "material.cpp"
    #include "matte.cpp"
    #include "glass.cpp"
    #include "mirror.cpp"
    #include "ai_material.cpp"
#else
    #include "material.h"
    #include "matte.h"
    #include "glass.h"
    #include "mirror.h"
    #include "ai_material.h"
#endif