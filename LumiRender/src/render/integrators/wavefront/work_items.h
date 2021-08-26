//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/macro.h"

namespace luminous {


    inline namespace render {
        template<typename T>
        struct SOA;
        struct Test {
            int a;
            int b;
        };

        template<>
        struct SOA<Test> {

        };
    }
}