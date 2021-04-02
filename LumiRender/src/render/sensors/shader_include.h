//
// Created by Zero on 2021/3/22.
//


#pragma once

// this file just for cuda shader include
#if defined(__CUDACC__)
    #include "sensor.cpp"
    #include "sensor_handle.cpp"
    #include "perspective_camera.cpp"
    #include "pinhole_camera.cpp"
#else
    #error "this file just for cuda shader include"
#endif

