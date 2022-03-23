#pragma once
#include "core/refl/type_reflection.h"
#include "render/textures/attr.h"

namespace luminous {
inline namespace render {

class MaterialConfig;

class ClothMaterial {

    DECLARE_REFLECTION(ClothMaterial)

public:
    ClothMaterial(const Attr3D &basecolor, const Attr1D &eta, const Attr1D &r, bool remapping_roughness = false);

    ClothMaterial(const MaterialConfig &config);

    LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

private:
    Attr3D _base_color; // Base layer color
    Attr1D _eta;         // Fresnel refraction factor
    Attr1D _roughness;  // Overlay(Specular) layer roughness
    bool _remapping_roughness; // Overlay(Specular) layer roughness remapping flag
};

LM_XPU_INLINE ClothMaterial::ClothMaterial(const Attr3D &basecolor, const Attr1D &eta, const Attr1D &r, bool remapping_roughness)
 : _base_color(basecolor), _eta(eta), _roughness(r), _remapping_roughness(remapping_roughness) {}

}
}