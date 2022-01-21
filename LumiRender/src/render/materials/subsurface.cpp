#if 0

#include "subsurface.h"
#include "parser/config.h"

namespace luminous::render {

BEGIN_MEMBER_MAP(SubsurfaceMaterial)
  MEMBER_MAP_ENTRY(_table.radius_samples)
  MEMBER_MAP_ENTRY(_table.rho_samples)
  MEMBER_MAP_ENTRY(_table.profile)
  MEMBER_MAP_ENTRY(_table.rhoEff)
  MEMBER_MAP_ENTRY(_table.profileCDF)
END_MEMBER_MAP()


SubsurfaceMaterial::SubsurfaceMaterial(const MaterialConfig &config)
  : SubsurfaceMaterial(config.sigma_a.tex_idx(), config.sigma_s.tex_idx(),
    config.reflectance.tex_idx(), config.mfp.tex_idx(), config.g, config.eta,
    config.uroughness.tex_idx(), config.vroughness.tex_idx(), config.normal_tex.tex_idx(),
    config.remapping_roughness
  ) {
}

SubsurfaceMaterial::SubsurfaceMaterial(
        index_t sigma_a, index_t sigma_s, index_t reflectance, index_t mfp,
        float g, float eta, index_t uroughness, index_t vroughness, index_t bump_map,
        bool remap_roughness) : _bump_map_idx(bump_map), _sigma_a_idx(sigma_a), _sigma_s_idx(sigma_s), _reflectance_idx(reflectance),
                                _mfp_idx(mfp), _eta(eta), _uroughness_idx(uroughness), _vroughness_idx(vroughness),
                                _remap_roughness(remap_roughness),
                                _table(100, 64) {
    ComputeBeamDiffusionBSSRDF(g, eta, &_table);
}

BSDFWrapper SubsurfaceMaterial::get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {

    return BSDFWrapper{};
}

TabulatedBSSRDF SubsurfaceMaterial::get_BSSRDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {
    return TabulatedBSSRDF{};
}


}// namespace luminous::render


#endif