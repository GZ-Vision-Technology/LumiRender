#ifndef __SUBSURFACE_H__
#define __SUBSURFACE_H__

#if 0

#include "render/scattering/bssrdf.h"
#include "base_libs/math/scalar_types.h"
#include "core/type_reflection.h"

namespace luminous {
inline namespace render {

struct MaterialConfig;
struct MaterialEvalContext;
struct SceneData;

class SubsurfaceMaterial {

    DECLARE_REFLECTION(SubsurfaceMaterial)

public:
#ifndef IS_GPU_CODE
    explicit SubsurfaceMaterial(const MaterialConfig &config);
#endif

    LM_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;
    LM_XPU TabulatedBSSRDF get_BSSRDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;
private:
#ifndef IS_GPU_CODE
    SubsurfaceMaterial(index_t sigma_a, index_t sigma_s, index_t reflectance, index_t mfp,
            float g, float eta, index_t uroughness, index_t vroughness, index_t bump_map,
            bool remap_roughness);
#endif

    index_t _bump_map_idx;
    index_t _sigma_a_idx, _sigma_s_idx, _reflectance_idx, _mfp_idx;
    float _scale;
    float _eta;
    index_t _uroughness_idx, _vroughness_idx;
    bool _remap_roughness;
    BSSRDFTable _table;

    DECLARE_MEMBER_MAP(SubsurfaceMaterial)
};

}
}

#endif



#endif // __SUBSURFACE_H__