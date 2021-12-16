//
// Created by Zero on 13/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "base.h"

namespace luminous {
    inline namespace render {

        template<typename TData, typename TFresnel, typename TMicrofacet, typename... TBxDF>
        class BSDF_Ty {
        protected:
            using Tuple = std::tuple<TBxDF...>;
            static constexpr int size = std::tuple_size_v<Tuple>;
            Tuple _bxdfs;
            TMicrofacet _microfacet{};
            TFresnel _fresnel{};
            TData _data{};
        protected:
            template<int index, typename F>
            LM_XPU void iterator(F &&func) const {
                if constexpr(index < size) {
                    auto obj = std::get<index>(_bxdfs);
                    if (func(obj)) {
                        iterator<index + 1>(func);
                    }
                }
            }

            template<int index, typename F>
            LM_XPU void iterator(F &&func) {
                if constexpr(index < size) {
                    auto obj = std::get<index>(_bxdfs);
                    if (func(obj)) {
                        iterator<index + 1>(func);
                    }
                }
            }

        public:
            LM_XPU BSDF_Ty() = default;

            LM_XPU explicit BSDF_Ty(TData data, TFresnel fresnel, TMicrofacet microfacet, TBxDF...args)
                    : _data(data), _fresnel(fresnel), _microfacet(microfacet),
                      _bxdfs(std::make_tuple(std::forward<TBxDF>(args)...)) {

            }

            LM_ND_XPU Spectrum color() const {
                return Spectrum{_data.color};
            }

            template<typename F>
            LM_XPU void for_each(F &&func) const {
                iterator<0>(std::move(func));
            }

            template<typename F>
            LM_XPU void for_each(F &&func) {
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
                this->for_each([&](auto bxdf) {
                    if (bxdf.match_flags(flags)) {
                        ret += bxdf.eval(wo, wi, _data, _fresnel, _microfacet);
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
                        ret += bxdf.PDF(wo, wi, _fresnel, _microfacet);
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

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                          BxDFFlags flags = BxDFFlags::All,
                                          TransportMode mode = TransportMode::Radiance) const {
                int num = match_num(flags);
                if (num == 0) {
                    return {};
                }

                int comp = std::min((int) std::floor(uc * num), num - 1);
                int count = 0;
                BSDFSample ret;
                for_each([&](auto bxdf) {
                    if (bxdf.match_flags(flags)) {
                        if (count == comp) {
                            ret = bxdf.sample_f(wo, u, _data, _fresnel, _microfacet, mode);
                            return false;
                        }
                        count += 1;
                    }
                    return true;
                });
                ret.PDF /= num;
                return ret;
            }

            LM_ND_XPU BxDFFlags flags() const {
                int ret{0};
                for_each([&](auto bxdf) {
                    ret |= bxdf.flags();
                    return true;
                });
                return static_cast<BxDFFlags>(ret);
            }
        };
    }
}