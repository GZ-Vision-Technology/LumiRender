//
// Created by Zero on 2021/5/5.
//

#pragma once

#if defined(__CUDACC__)
    #include "interaction.cpp"
#else
    #include "interaction.h"
#endif
