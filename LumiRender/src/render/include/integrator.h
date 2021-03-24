//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"

namespace luminous {
    class Integrator : public Noncopyable {
    public:
        virtual ~Integrator() {}

        virtual void render() = 0;
    };
}