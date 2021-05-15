//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "graphics/math/common.h"
#include <embree3/rtcore.h>
#include "core/concepts.h"

namespace luminous {
    inline namespace cpu {
        class EmbreeAccel : public Noncopyable {
        private:
            RTCDevice _rtc_device{nullptr};
        public:

        };
    } // luminous::cpu
} // luminous