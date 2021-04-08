////
//// Created by Zero on 2021/4/7.
////
//
//
//#pragma once
//
//#include "light.h"
//#include "render/include/scene_graph.h"
//
//namespace luminous {
//    inline namespace render {
//
//        class SpotLight : public LightBase {
//        private:
//            float3 _pos;
//            float3 _axis;
//            float _cos_theta_i;
//            float _cos_theta_o;
//            float3 _intensity;
//        public:
//            SpotLight(float3 pos, float3 intensity, float theta_i, float theta_o)
//                    : LightBase(LightType::DeltaPosition),
//                      _pos(pos),
//                      _intensity(intensity),
//                      _cos_theta_i(cos(radians(theta_i))),
//                      _cos_theta_o(cos(radians(theta_o))) {}
//
//            GEN_CLASS_NAME(SpotLight)
//
//            NDSC_XPU Interaction sample(float u, const HitGroupData *hit_group_data) const;
//
//            NDSC_XPU LightLiSample Li(LightLiSample lls) const;
//
//            NDSC_XPU float PDF_Li(const Interaction &ref_p, const Interaction &p_light) const;
//
//            NDSC_XPU float3 power() const;
//
//            NDSC std::string to_string() const;
//
//            static SpotLight create(const LightConfig &config);
//        };
//
//    } // luminous::render
//} // luminous