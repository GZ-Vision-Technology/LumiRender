//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/geometry/common.h"
#include "base_libs/optics/common.h"
#include "render/include/config.h"
#include "core/refl/reflection.h"
#include "light_util.h"

namespace luminous {
    inline namespace render {

        struct SceneData;

        class LightBase : BASE_CLASS() {
        public:
            REFL_CLASS(LightBase)
        protected:
            const LightType _type;
        public:
            XPU explicit LightBase(LightType type)
                    : _type(type) {}

            NDSC_XPU LightType type() const {
                return _type;
            }

            NDSC_XPU Spectrum on_miss(Ray ray, const SceneData *data) const {
                return {0.f};
            }

            NDSC_XPU bool is_infinite() const {
                return _type == LightType::Infinite;
            }

            NDSC_XPU bool is_delta() const {
                return _type == LightType::DeltaDirection || _type == LightType::DeltaPosition;
            }

            GEN_STRING_FUNC({
                                return string_printf("type : %d", (int) _type);
                            })
        };
    }
}