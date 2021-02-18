//
// Created by Zero on 2021/1/15.
//


#pragma once

#include "graphics/optics/common.h"
#include "graphics/geometry/common.h"
#include "core/concepts.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    namespace render {

        class PinholeCamera;
        class PerspectiveCamera;


        using lstd::Variant;
        class SensorHandle : public Variant<PinholeCamera*, PerspectiveCamera*> {
        public:
            using Variant::Variant;

            SensorHandle(void *) {}
        };


    }
}
