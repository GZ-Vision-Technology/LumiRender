//
// Created by Zero on 2021/1/6.
//


#pragma once

#include "buffer.h"

namespace luminous::backend {
    class Acceleration {
    public:
        virtual void intersect_closet() const = 0;
    };
}