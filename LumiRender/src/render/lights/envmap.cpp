//
// Created by Zero on 2021/7/28.
//

#include "envmap.h"



namespace luminous {
    inline namespace render {

        void Envmap::preprocess(const Scene *scene) {

        }

        LightLiSample Envmap::Li(LightLiSample lls) const {
            return LightLiSample();
        }

        SurfaceInteraction Envmap::sample(float2 u, const HitGroupData *hit_group_data) const {
            return SurfaceInteraction();
        }

        float Envmap::PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light) const {
            return 0;
        }

        Spectrum Envmap::power() const {
            // todo
            return luminous::Spectrum(1.f);
        }

        void Envmap::print() const {
            printf("type:Envmap\n");
        }

        CPU_ONLY(Envmap Envmap::create(const LightConfig &config) {
            return Envmap(*config.tex, config.o2w_config.create(), config.distribution);
        })
    }
}