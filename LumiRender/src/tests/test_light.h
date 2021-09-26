//
// Created by Zero on 26/09/2021.
//


#pragma once

#include "render/lights/light.h"

namespace luminous {
    inline namespace render {
        class LB : BASE_CLASS() {
        public:
            REFL_CLASS(LB)
        public:
            const LightType _type{};
        public:
            XPU explicit LB(LightType type)
            : _type(type) {}
        };

        class AL : public LB , public luminous::ICreator<AL> {
        public:
            float padded{};
            AL(): LB(LightType::Area) {}
        };
    }
}