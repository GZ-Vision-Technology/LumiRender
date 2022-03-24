//
// Created by Zero on 2021/4/30.
//


#pragma once


#include "base_libs/lstd/variant.h"
#include "core/refl/type_reflection.h"
#include "render/include/interaction.h"
#include "parser/config.h"

namespace luminous {
    inline namespace render {
        class MatteMaterial;

        class MirrorMaterial;

        class GlassMaterial;

        class FakeMetalMaterial;

        class MetalMaterial;

        using lstd::Variant;

        class DisneyMaterial;

        class SubstrateMaterial;

        class ClothMaterial;

        class Material : public Variant<MatteMaterial *, MirrorMaterial *, SubstrateMaterial *,
                GlassMaterial *, FakeMetalMaterial *, MetalMaterial *, DisneyMaterial *, ClothMaterial *> {

        DECLARE_REFLECTION(Material, Variant)

        protected:
            index_t _normal_idx{invalid_uint32};

        public:
            using Variant::Variant;

            LM_ND_XPU MaterialEvalContext
            compute_shading_frame(MaterialEvalContext ctx, const SceneData *scene_data) const;

            LM_ND_XPU BSDFWrapper get_BSDF(MaterialEvalContext ctx, const SceneData *scene_data) const;

#ifndef __CUDACC__

            LM_NODISCARD static std::pair<Material, std::vector<size_t>> create(const MaterialConfig &mc);

#endif
        };

    } // luminous::render
} // luminous

