//
// Created by Zero on 2021/3/22.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "camera_base.cpp"
    #include "sensor.cpp"
    #include "thin_lens_camera.cpp"
    #include "film.cpp"
#else
    #include "camera_base.h"
    #include "sensor.h"
    #include "thin_lens_camera.h"
    #include "film.h"
#endif

