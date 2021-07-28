//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "graphics/math/common.h"
#include "core/backend/buffer_view.h"


namespace luminous {
    inline namespace render {
        using std::vector;
        using std::move;
        struct Distribution1DBuilder {
        public:
            std::vector<float> func;
            std::vector<float> CDF;
            float func_integral;

            Distribution1DBuilder() = default;

            Distribution1DBuilder(std::vector<float> func, std::vector<float> CDF, float integral)
                    : func(move(func)), CDF(move(CDF)), func_integral(integral) {}
        };

        class Distribution1D {
        public:
            using value_type = float;
            using const_value_type = const float;
        private:
            // todo change to indice mode, reduce memory usage
            BufferView <const_value_type> _func;
            BufferView <const_value_type> _CDF;
            float _func_integral;
        public:
            XPU Distribution1D(BufferView <const_value_type> func,
                               BufferView <const_value_type> CDF, float integral)
                    : _func(func), _CDF(CDF), _func_integral(integral) {}

            NDSC_XPU size_t size() const { return _func.size(); }

            NDSC_XPU float sample_continuous(float u, float *pdf = nullptr, int *off = nullptr) const {
                auto predicate = [&](int index) {
                    return _CDF[index] <= u;
                };
                int offset = find_interval((int) _CDF.size(), predicate);
                if (off) {
                    *off = offset;
                }
                float du = u - _CDF[offset];
                if ((_CDF[offset + 1] - _CDF[offset]) > 0) {
                    DCHECK_GT(_CDF[offset + 1], _CDF[offset]);
                    du /= (_CDF[offset + 1] - _CDF[offset]);
                }
                DCHECK(!is_nan(du));

                if (pdf) {
                    *pdf = (_func_integral > 0) ? _func[offset] / _func_integral : 0;
                }
                return (offset + du) / size();
            }

            NDSC_XPU int sample_discrete(float u, float *PMF = nullptr, float *u_remapped = nullptr) const {
                auto predicate = [&](int index) {
                    return _CDF[index] <= u;
                };
                int offset = find_interval(_CDF.size(), predicate);
                if (PMF) {
                    *PMF = (_func_integral > 0) ? _func[offset] / (_func_integral * size()) : 0;
                }
                if (u_remapped) {
                    *u_remapped = (u - _CDF[offset]) / (_CDF[offset + 1] - _CDF[offset]);
                    DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                }
                return offset;
            }

            NDSC_XPU float integral() const { return _func_integral; }

            template<typename Index>
            NDSC_XPU float func_at(Index i) const { return _func[i]; }

            template<typename Index>
            NDSC_XPU float PMF(Index i) const {
                DCHECK(i >= 0 && i < size());
                return func_at(i) / (integral() * size());
            }

            static Distribution1DBuilder create_builder(std::vector<float> func) {
                size_t num = func.size();
                std::vector<float> CDF(num + 1);
                CDF[0] = 0;
                for (int i = 1; i < num + 1; ++i) {
                    CDF[i] = CDF[i - 1] + func[i - 1] / num;
                }
                float integral = CDF[num];
                if (integral == 0) {
                    for (int i = 1; i < num + 1; ++i) {
                        CDF[i] = float(i) / float(num);
                    }
                } else {
                    for (int i = 1; i < num + 1; ++i) {
                        CDF[i] = CDF[i] / integral;
                    }
                }
                return Distribution1DBuilder(move(func), move(CDF), integral);
            }
        };

        struct Distribution2DBuilder {
            vector<Distribution1DBuilder> conditional_v;
            Distribution1DBuilder marginal;
            Distribution2DBuilder(vector<Distribution1DBuilder> conditional_v, Distribution1DBuilder marginal)
                : conditional_v(move(conditional_v)), marginal(move(marginal)) {}
        };


        class Distribution2D {
        private:
            BufferView <Distribution1D> _conditional_v;
            Distribution1D _marginal;
        public:
            XPU Distribution2D(BufferView <Distribution1D> conditional_v, Distribution1D marginal)
                    : _conditional_v(conditional_v), _marginal(marginal) {}

            NDSC_XPU float2 sample_continuous(float2 u, float *PDF) {
                float PDFs[2];
                int v;
                float d1 = _marginal.sample_continuous(u[1], &PDFs[1], &v);
                float d0 = _conditional_v[v].sample_continuous(u[0], &PDFs[0]);
                *PDF = PDFs[0] * PDFs[1];
                return make_float2(d0, d1);
            }

            NDSC_XPU float PDF(float2 p) const {
                int iu = clamp(int(p[0] * _conditional_v[0].size()), 0, _conditional_v[0].size());
                int iv = clamp(int(p[1] * _marginal.size()), 0, _marginal.size());
                return _conditional_v[iv].func_at(iu) / _marginal.integral();
            }

            static Distribution2DBuilder create_builder(float *func, int nu, int nv) {
                vector<Distribution1DBuilder> conditional_v;
                conditional_v.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    vector<float> func_v;
                    func_v.insert(func_v.end(), &func[v * nu], &func[v * nu + nv]);
                    Distribution1DBuilder builder = Distribution1D::create_builder(func_v);
                    conditional_v.push_back(builder);
                }
                vector<float> marginal_func;
                marginal_func.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    marginal_func.push_back(conditional_v[v].func_integral);
                }
                Distribution1DBuilder marginal_builder = Distribution1D::create_builder(marginal_func);
                return Distribution2DBuilder(move(conditional_v), move(marginal_builder));
            }
        };
    } // luminous::sampling
} // luminous