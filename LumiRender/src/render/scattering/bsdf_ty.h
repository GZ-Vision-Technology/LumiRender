//
// Created by Zero on 13/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "base_libs/sampling/sampling.h"
#include "microfacet.h"
#include "bsdf_data.h"
#include "base.h"

namespace luminous {
    inline namespace render {

        template<typename TData, bool Uniform, typename... TBxDF>
        class BSDF_Ty {
        protected:
            using Tuple = std::tuple<TBxDF...>;
            static constexpr int size = std::tuple_size_v<Tuple>;
            Tuple _bxdfs{};
            TData _data{};
        protected:
            template<int index, typename F>
            LM_XPU void iterator(F &&func) const {
                if constexpr(index < size) {
                    auto obj = std::get<index>(_bxdfs);
                    if (func(obj, index)) {
                        iterator<index + 1>(func);
                    }
                }
            }

            template<int index, typename F>
            LM_XPU void iterator(F &&func) {
                if constexpr(index < size) {
                    auto obj = std::get<index>(_bxdfs);
                    if (func(obj, index)) {
                        iterator<index + 1>(func);
                    }
                }
            }

        public:
            LM_XPU BSDF_Ty() : _bxdfs(TBxDF()...) {

            }


            LM_XPU explicit BSDF_Ty(TData data, TBxDF...args)
                    : _data(data),
                      _bxdfs(std::make_tuple(std::forward<TBxDF>(args)...)) {

            }

            LM_ND_XPU Spectrum color() const {
                return std::get<0>(_bxdfs).spectrum();
            }

            template<typename T, int index>
            LM_XPU void fill_BxDF(T bxdf) {
                using Type = std::tuple_element_t<index, Tuple>;
                if constexpr(std::is_same_v<Type, T>) {
                    std::get<index>(_bxdfs) = bxdf;
                } else if constexpr(index < size) {
                    fill_BxDF<T, index + 1>(bxdf);
                } else {
                    static_assert(index < size, "!");
                }
            }

            template<typename T>
            LM_XPU void add_BxDF(T bxdf) {
                // todo performance
                fill_BxDF<T, 0>(bxdf);
            }

            template<typename T>
            LM_XPU void set_data(T data) {
                _data = data;
            }

            template<typename F>
            LM_XPU void for_each(F &&func) const {
                iterator<0>(std::move(func));
            }

            template<typename F>
            LM_XPU void for_each(F &&func) {
                iterator<0>(std::move(func));
            }

            LM_ND_XPU int match_num(BxDFFlags bxdf_flags) const {
                int ret{0};
                for_each([&](auto bxdf, int idx) {
                    if (bxdf.match_flags(bxdf_flags)) {
                        ret += 1;
                    }
                    return true;
                });
                return ret;
            }

            LM_ND_XPU BxDFFlags combine_flags(float3 wo, float3 wi, BxDFFlags flags) const {
                bool reflect = same_hemisphere(wo, wi);
                auto non_reflect = ~BxDFFlags::Reflection;
                auto non_trans = ~BxDFFlags::Transmission;

                flags = static_cast<BxDFFlags>(reflect ?
                                               flags & non_trans :
                                               flags & non_reflect);
                return flags;
            }

            LM_ND_XPU BxDFFlags flags() const {
                int ret{0};
                for_each([&](auto bxdf, int idx) {
                    ret |= bxdf.flags();
                    return true;
                });
                return static_cast<BxDFFlags>(ret);
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BxDFFlags flags = BxDFFlags::All,
                                    TransportMode mode = TransportMode::Radiance) const {
                if (wo.z == 0) {
                    return {0.f};
                }

                flags = combine_flags(wo, wi, flags);

                Spectrum ret{0.f};
                this->for_each([&](auto bxdf, int idx) {
                    if (bxdf.match_flags(flags)) {
                        auto tmp = bxdf.safe_eval(wo, wi, _data.get_helper());
                        DCHECK(!has_invalid(tmp));
                        ret += tmp;
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

                flags = combine_flags(wo, wi, flags);

                float ret{0.f};
                for_each([&](auto bxdf, int idx) {
                    if (bxdf.match_flags(flags)) {
                        match_count += 1;
                        ret += bxdf.safe_PDF(wo, wi, _data.get_helper());
                    }
                    return true;
                });
                return match_count > 0 ? ret / match_count : 0;
            }

            LM_ND_XPU BSDFSample uniform_sample_f(float3 wo, float uc, float2 u,
                                                  BxDFFlags flags = BxDFFlags::All,
                                                  TransportMode mode = TransportMode::Radiance) const {
                int num = match_num(flags);
                if (num == 0) {
                    return {};
                }
                BSDFHelper helper = _data;
                int comp = std::min((int) std::floor(uc * num), num - 1);
                uc = remapping(uc, float(comp) / num, float(comp + 1) / num);
                int count = 0;
                BSDFSample ret;
                for_each([&](auto bxdf, int idx) {
                    if (bxdf.match_flags(flags)) {
                        if (count == comp) {
                            ret = bxdf.sample_f(wo, uc, u, _data.get_helper(), mode);
                            ret.albedo = bxdf.color(helper);
                            return false;
                        }
                        count += 1;
                    }
                    return true;
                });
                ret.PDF /= num;

                return ret;
            }

            LM_ND_XPU BSDFSample importance_sample_f(float3 wo, float uc, float2 u,
                                                     BxDFFlags flags = BxDFFlags::All,
                                                     TransportMode mode = TransportMode::Radiance) const {
                static constexpr float weight_threshold = 0.02;
                float weights[size] = {};
                BSDFHelper helper = _data;
                helper.correct_eta(Frame::cos_theta(wo));
                Spectrum fr = helper.eval_fresnel(Frame::abs_cos_theta(wo));
                for_each([&](auto bxdf, int index) {
                    float weight = bxdf.weight(helper, fr);
                    weights[index] = weight;
                    return true;
                });

                BufferView<const float> bfv(weights);
                float PMF = 1;
                int comp = sample_discrete(bfv, uc, &PMF, &uc);
                BSDFSample ret;
                for_each([&](auto bxdf, int index) {
                    if (index == comp) {
                        ret = bxdf.sample_f(wo, uc, u, _data.get_helper(), mode);
                        ret.albedo = bxdf.color(helper);
                        return false;
                    } else {
                        return true;
                    }
                });
                ret.PDF *= PMF;
                return ret;
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                          BxDFFlags flags = BxDFFlags::All,
                                          TransportMode mode = TransportMode::Radiance) const {
                if constexpr(Uniform) {
                    return uniform_sample_f(wo, uc, u, flags, mode);
                } else {
                    return importance_sample_f(wo, uc, u, flags, mode);
                }
            }
        };

        template<typename TData, typename T1, typename T2, bool Delta = false>
        class FresnelBSDF : public BSDF_Ty<TData, false, T1, T2> {
        protected:
            using BaseClass = BSDF_Ty<TData,false, T1, T2>;
            static_assert(BaseClass::size == 2, "FresnelBSDF must be two BxDF lobe!");
            using Refl = std::tuple_element_t<0, typename BaseClass::Tuple>;
            using Trans = std::tuple_element_t<1, typename BaseClass::Tuple>;
        public:
            using BaseClass::BaseClass;

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BxDFFlags flags = BxDFFlags::All,
                                    TransportMode mode = TransportMode::Radiance) const {
                if constexpr(Delta) {
                    return {0.f};
                } else {
                    BSDFHelper helper = BaseClass::_data.get_helper();
                    float cos_theta_o = Frame::cos_theta(wo);
                    helper.correct_eta(cos_theta_o);
                    if (same_hemisphere(wi, wo)) {
                        return std::get<0>(BaseClass::_bxdfs).eval(wo, wi, helper, mode);
                    }
                    return std::get<1>(BaseClass::_bxdfs).eval(wo, wi, helper, mode);
                }
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BxDFFlags flags = BxDFFlags::All,
                                TransportMode mode = TransportMode::Radiance) const {
                if constexpr(Delta) {
                    return 0.f;
                } else {
                    BSDFHelper helper = BaseClass::_data.get_helper();
                    float cos_theta_o = Frame::cos_theta(wo);
                    helper.correct_eta(cos_theta_o);
                    if (same_hemisphere(wi, wo)) {
                        return std::get<0>(BaseClass::_bxdfs).PDF(wo, wi, helper, mode);
                    }
                    return std::get<1>(BaseClass::_bxdfs).PDF(wo, wi, helper, mode);
                }
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                          BxDFFlags flags = BxDFFlags::All,
                                          TransportMode mode = TransportMode::Radiance) const {
                int num = BaseClass::match_num(flags);
                if (num == 0) {
                    return {};
                }

                BSDFHelper helper = BaseClass::_data.get_helper();

                float cos_theta_o = Frame::cos_theta(wo);
                helper.correct_eta(cos_theta_o);
                float Fr = helper.eval_fresnel(Frame::abs_cos_theta(wo))[0];
                BSDFSample ret;
                if (uc < Fr) {
                    ret = std::get<0>(BaseClass::_bxdfs)._sample_f(wo, uc, u, Fr, helper, mode);
                    ret.PDF *= Fr;
                } else {
                    ret = std::get<1>(BaseClass::_bxdfs)._sample_f(wo, uc, u, Fr, helper, mode);
                    ret.PDF *= 1 - Fr;
                }
                ret.albedo = std::get<0>(BaseClass::_bxdfs).color(helper);
                return ret;
            }
        };
    }
}