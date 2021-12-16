//
// Created by Zero on 13/12/2021.
//


#pragma once

#include "base_libs/lstd/variant.h"
#include "base_libs/optics/rgb.h"
#include "base.h"

namespace luminous {
    inline namespace render {

        template<typename TMicrofacet, typename TFresnel, typename... Ts>
        class BSDF {
        protected:
            using Tuple = std::tuple<Ts...>;
            static constexpr int size = std::tuple_size_v<Tuple>;
            Tuple _bxdfs;
            TMicrofacet _microfacet;
            TFresnel _fresnel;

        protected:
            template<int index, typename F>
            void iterator(F &&func) {
                if constexpr(index < size) {
                    auto obj = std::get<index>(_bxdfs);

                    if (func(obj)) {
                        iterator<index + 1>(func);
                    }
                }
            }

        public:
            BSDF() = default;

            explicit BSDF(Ts...args) {
                _bxdfs = std::make_tuple(std::forward<Ts>(args)...);
            }

            template<typename F>
            void for_each(F &&func) {
                iterator<0>(std::move(func));
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BxDFFlags flags = BxDFFlags::All,
                                    TransportMode mode = TransportMode::Radiance) const {
                if (wo.z == 0) {
                    return {0.f};
                }
                bool reflect = same_hemisphere(wo, wi);
                auto non_reflect = ~BxDFFlags::Reflection;
                auto non_trans = ~BxDFFlags::Transmission;

                flags = static_cast<BxDFFlags>(reflect ?
                                               flags & non_trans :
                                               flags & non_reflect);

                Spectrum ret{0.f};
                for_each([&](auto bxdf) {
                    if (bxdf.match_flags(flags)) {
                        ret += bxdf.eval(wo, wi, mode);
                    }
                    return true;
                });
                return ret;
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BxDFFlags flags = BxDFFlags::All,
                                TransportMode mode = TransportMode::Radiance) const {
                if (wo.z == 0) {
                    return 0;
                }
                int match_count = 0;
                float ret{0.f};
                for_each([&](auto bxdf) {
                    if (bxdf.match_flags(flags)) {
                        match_count += 1;
                        ret += bxdf.PDF(wo, wi);
                    }
                    return true;
                });
                return match_count > 0 ? ret / match_count : 0;
            }

            LM_ND_XPU int match_num(BxDFFlags bxdf_flags) const {
                int ret{0};
                for_each([&](auto bxdf) {
                    if (bxdf.match_flags(bxdf_flags)) {
                        ret += 1;
                    }
                    return true;
                });
                return ret;
            }

            LM_ND_XPU lstd::optional<BSDFSample> sample_f(float3 wo, float uc, float2 u,
                                                          BxDFFlags flags = BxDFFlags::All,
                                                          TransportMode mode = TransportMode::Radiance) const {
                int num = match_num(flags);
                if (num == 0) {
                    return {};
                }

                int comp = std::min((int)std::floor(uc * num), num - 1);
                int count = 0;
                lstd::optional<BSDFSample> ret;
                for_each([&](auto bxdf) {
                    if (bxdf.match_flags(flags)) {
                        if (++count == comp) {
                            ret = bxdf.sample_f(wo, uc, u, mode);
                            return false;
                        }
                    }
                    return true;
                });
                ret->PDF /= num;
                return ret;
            }

            LM_ND_XPU BxDFFlags flags() const {
                std::byte ret{0};
                for_each([&](auto bxdf) {
                    ret |= bxdf.flags();
                    return true;
                });
                return static_cast<BxDFFlags>(ret);
            }
        };
    }
}