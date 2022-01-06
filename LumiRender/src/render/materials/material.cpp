//
// Created by Zero on 26/09/2021.
//

#include "material.h"
#include "common.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        BSDFWrapper Material::get_BSDF(MaterialEvalContext ctx, const SceneData *scene_data) const {
            ctx = compute_shading_frame(ctx, scene_data);
            LUMINOUS_VAR_PTR_DISPATCH(get_BSDF, ctx, scene_data)
        }

        MaterialEvalContext Material::compute_shading_frame(MaterialEvalContext ctx,
                                                            const SceneData *scene_data) const {
            if (!is_valid_index(_normal_idx)) {
                return ctx;
            }
            float3 normal = make_float3(scene_data->get_texture(_normal_idx).eval(ctx)) * 2.f - make_float3(1.f);
            float scale = 1.f;
            normal.x *= scale;
            normal.y *= scale;

            float2 v2 = make_float2(normal);
            normal.z = safe_sqrt(1 - length_squared(v2));

            Frame shading_frame = Frame::from_xz(ctx.dp_dus, ctx.ns);

            ctx.ns = shading_frame.to_world(normal);
            float3 dp_dvs = cross(ctx.ns, ctx.dp_dus);
            ctx.dp_dus = cross(ctx.ns, dp_dvs);
            return ctx;
        }

#ifndef __CUDACC__
        std::pair<Material, std::vector<size_t>> Material::create(const MaterialConfig &mc) {
            auto ret = detail::create_ptr<Material>(mc);
            ret.first._normal_idx = mc.normal_tex.tex_idx();
            return ret;
        }
#endif
    }
}