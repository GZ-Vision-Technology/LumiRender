//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/geometry/common.h"
#include "base_libs/optics/common.h"
#include "parser/config.h"
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
            LM_XPU explicit LightBase(LightType type)
                    : _type(type) {}

            LM_ND_XPU LightType type() const {
                return _type;
            }

            LM_ND_XPU Spectrum on_miss(float3 dir, const SceneData *data) const {
                return {0.f};
            }

            LM_ND_XPU bool is_infinite() const {
                return _type == LightType::Infinite;
            }

            LM_ND_XPU bool is_delta() const {
                return _type == LightType::DeltaDirection || _type == LightType::DeltaPosition;
            }

            GEN_STRING_FUNC({
                                return string_printf("type : %d", (int) _type);
                            })
        };
    }
}