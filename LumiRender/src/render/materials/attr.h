//
// Created by Zero on 19/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/scene/scene_data.h"
#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {

        struct TextureHandle {
        protected:
            index_t _tex_idx{invalid_uint32};
        public:
            LM_XPU explicit TextureHandle(index_t idx = invalid_uint32) : _tex_idx(idx) {}

            ND_XPU_INLINE bool tex_valid() const {
                return _tex_idx != invalid_uint32;
            }

            ND_XPU_INLINE const Texture &get_texture(const SceneData *scene_data) const {
                return scene_data->get_texture(_tex_idx);
            }

            ND_XPU_INLINE float4 eval_tex(const SceneData *scene_data, const MaterialEvalContext &ctx) const {
                return get_texture(scene_data).eval(ctx);
            }
        };

        struct Attr1D : TextureHandle {
        protected:
            float _val{0.f};
        public:
            LM_XPU explicit Attr1D(float val = 0.f) : TextureHandle(), _val(val) {}

            LM_XPU explicit Attr1D(index_t idx) : TextureHandle(idx), _val(0.f) {}

            ND_XPU_INLINE float eval(const SceneData *scene_data, const MaterialEvalContext &ctx) const {
                return tex_valid() ? eval_tex(scene_data, ctx).x : _val;
            }
        };

        struct Attr2D : TextureHandle {
        protected:
            float2 _val{};
        public:
            LM_XPU explicit Attr2D(float2 val = {}) : TextureHandle(), _val(val) {}

            LM_XPU explicit Attr2D(index_t idx) : TextureHandle(idx), _val({}) {}

            ND_XPU_INLINE float2 eval(const SceneData *scene_data, const MaterialEvalContext &ctx) const {
                return tex_valid() ? make_float2(eval_tex(scene_data, ctx)) : _val;
            }
        };

        struct Attr3D : TextureHandle {
        protected:
            float _val0{};
            float _val1{};
            float _val2{};
        public:
            LM_XPU explicit Attr3D(float3 val = {})
                    : _val0(val.x), _val1(val.y), _val2(val.z),
                      TextureHandle() {}

            ND_XPU_INLINE float3 value() const { return make_float3(_val0, _val1, _val2); }

            LM_XPU explicit Attr3D(index_t idx) : TextureHandle(idx) {}

            ND_XPU_INLINE float3 eval(const SceneData *scene_data, const MaterialEvalContext &ctx) const {
                return tex_valid() ? make_float3(eval_tex(scene_data, ctx)) : value();
            }
        };

        struct Attr4D : TextureHandle {
        protected:
            float4 _val{};
        public:
            LM_XPU explicit Attr4D(float4 val = {}) : TextureHandle(), _val(val) {}

            LM_XPU explicit Attr4D(index_t idx) : TextureHandle(idx), _val({}) {}

            ND_XPU_INLINE float4 eval(const SceneData *scene_data, const MaterialEvalContext &ctx) const {
                return tex_valid() ? eval_tex(scene_data, ctx) : _val;
            }
        };
    }
}