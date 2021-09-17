//
// Created by Zero on 17/09/2021.
//


#pragma once

#include "base_libs/math/interval.h"
#include <vector>


namespace luminous {
    inline namespace refl {
        struct Node {
            int interval_index{-1};
            int start{}, end{};
        };

        class PtrMapper {
        private:
            std::vector<PtrInterval> _intervals;

        };
    }
}