//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/lstd/common.h"
#include "alias_table.h"
#include <vector>

namespace luminous {
    inline namespace sampling {
        using std::vector;
        using std::move;

        struct DichotomyBuilder {
        public:
            std::vector<float> func;
            std::vector<float> CDF;
            float func_integral{};

            DichotomyBuilder() = default;

            DichotomyBuilder(std::vector<float> func, std::vector<float> CDF, float integral)
                    : func(move(func)), CDF(move(CDF)), func_integral(integral) {}
        };

        struct DichotomyData {
        public:
            using value_type = float;
            using const_value_type = const float;
            using Builder = DichotomyBuilder;
        public:
            // todo change to indice mode, reduce memory usage
            BufferView<const_value_type> func{};
            BufferView<const_value_type> CDF{};
            float func_integral{};

            DichotomyData() = default;

            DichotomyData(BufferView<const_value_type> func,
                          BufferView<const_value_type> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}
        };

        template<int Size>
        struct StaticDichotomyData {
        public:
            using Builder = DichotomyBuilder;
        private:
            static constexpr int size_in_bytes = Size * sizeof(float);
        public:
            Array<float, Size> func;
            Array<float, Size + 1> CDF;
            float func_integral{};

            StaticDichotomyData() = default;

            StaticDichotomyData(Array<float, Size> func,
                                Array<float, Size + 1> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}

            StaticDichotomyData(const float *f, const float *C, float integral) {
                init(f, C, integral);
            }

            explicit StaticDichotomyData(const Builder &builder) {
                init(builder.func.data(), builder.CDF.data(), builder.func_integral);
            }

            void init(const float *f, const float *C, float integral) {
                std::memcpy(func.begin(), f, size_in_bytes);
                std::memcpy(CDF.begin(), C, size_in_bytes + sizeof(float));
                func_integral = integral;
            }
        };

        template<typename T = DichotomyData>
        class TDichotomySampler {
        public:
            using Builder = DichotomyBuilder;
            using data_type = T;
        private:
            data_type _data;
        public:
            TDichotomySampler() = default;

            explicit TDichotomySampler(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TDichotomySampler(Args ...args) : TDichotomySampler(T(std::forward<Args>(args)...)) {}

            LM_ND_XPU size_t size() const { return _data.func.size(); }

            LM_ND_XPU float sample_continuous(float u, float *pdf, int *ofs) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                size_t offset = find_interval((int) _data.CDF.size(), predicate);
                *ofs = offset;
                float du = u - _data.CDF[offset];
                if ((_data.CDF[offset + 1] - _data.CDF[offset]) > 0) {
                    DCHECK_GT(_data.CDF[offset + 1], _data.CDF[offset]);
                    du /= (_data.CDF[offset + 1] - _data.CDF[offset]);
                }
                DCHECK(!is_nan(du));

                *pdf = (_data.func_integral > 0) ? _data.func[offset] / _data.func_integral : 0;
                return (offset + du) / size();
            }

            LM_ND_XPU int sample_discrete(float u, float *p, float *u_remapped) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                int offset = find_interval(_data.CDF.size(), predicate);
                *p = PMF(offset);
                *u_remapped = (u - _data.CDF[offset]) / (_data.CDF[offset + 1] - _data.CDF[offset]);
                DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                return offset;
            }

            LM_ND_XPU float integral() const { return _data.func_integral; }

            template<typename Index>
            LM_ND_XPU float func_at(Index i) const { return _data.func[i]; }

            template<typename Index>
            LM_ND_XPU float PMF(Index i) const {
                DCHECK(i >= 0 && i < size());
                return integral() > 0 ? (func_at(i) / (integral() * size())) : 0;
            }

            static DichotomyBuilder create_builder(std::vector<float> func) {
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
                return {move(func), move(CDF), integral};
            }
        };

        using DichotomySampler = TDichotomySampler<DichotomyData>;

        template<int Size>
        using StaticDichotomySampler = TDichotomySampler<StaticDichotomyData<Size>>;

        struct Dichotomy2DBuilder {
            vector<DichotomyBuilder> conditional_v;
            DichotomyBuilder marginal;

            Dichotomy2DBuilder(vector<DichotomyBuilder> conditional_v, DichotomyBuilder marginal)
                    : conditional_v(move(conditional_v)), marginal(move(marginal)) {}
        };

        struct Dichotomy2DData {
        public:
            BufferView<const DichotomySampler> conditional_v{};
            DichotomySampler marginal{};

            Dichotomy2DData() = default;

            Dichotomy2DData(BufferView<const DichotomySampler> conditional_v,
                            DichotomySampler marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<int U, int V>
        struct StaticDichotomy2DData {
        public:
            Array<StaticDichotomySampler<U>, V> conditional_v{};
            StaticDichotomySampler<V> marginal;

            StaticDichotomy2DData() = default;

            StaticDichotomy2DData(Array<StaticDichotomySampler<U>, V> conditional_v,
                                  StaticDichotomySampler<V> marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<typename T>
        class TDichotomy2D {
        public:
            using data_type = T;
        private:
            data_type _data;
        public:
            TDichotomy2D() = default;

            explicit TDichotomy2D(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TDichotomy2D(Args ...args) : TDichotomy2D(T(std::forward<Args>(args)...)) {}

            LM_ND_XPU float2 sample_continuous(float2 u, float *PDF, int2 *offset) const {
                float PDFs[2];
                int2 uv;
                float d1 = _data.marginal.sample_continuous(u[1], &PDFs[1], &uv[1]);
                float d0 = _data.conditional_v[uv[1]].sample_continuous(u[0], &PDFs[0], &uv[0]);
                *PDF = PDFs[0] * PDFs[1];
                *offset = uv;
                return make_float2(d0, d1);
            }

            LM_ND_XPU float func_at(int2 coord) const {
                auto row = _data.conditional_v[coord.y];
                return row.func_at(coord.x);
            }

            LM_ND_XPU float PDF(float2 p) const {
                size_t iu = clamp(size_t(p[0] * _data.conditional_v[0].size()), 0, _data.conditional_v[0].size() - 1);
                size_t iv = clamp(size_t(p[1] * _data.marginal.size()), 0, _data.marginal.size() - 1);
                return _data.conditional_v[iv].func_at(iu) / _data.marginal.integral();
            }

            LM_ND_XPU float integral() const {
                return _data.marginal.integral();
            }

            static Dichotomy2DBuilder create_builder(const float *func, int nu, int nv) {
                vector<DichotomyBuilder> conditional_v;
                conditional_v.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    vector<float> func_v;
                    func_v.insert(func_v.end(), &func[v * nu], &func[v * nu + nu]);
                    DichotomyBuilder builder = DichotomySampler::create_builder(func_v);
                    conditional_v.push_back(builder);
                }
                vector<float> marginal_func;
                marginal_func.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    marginal_func.push_back(conditional_v[v].func_integral);
                }
                DichotomyBuilder marginal_builder = DichotomySampler::create_builder(marginal_func);
                return {move(conditional_v), move(marginal_builder)};
            }
        };

        using Dichotomy2D = TDichotomy2D<Dichotomy2DData>;

        template<int U, int V>
        using StaticDichotomy2D = TDichotomy2D<StaticDichotomy2DData<U, V>>;

        template<int U, int V>
        LM_NODISCARD StaticDichotomy2D<U, V> create_static_distrib2d_old(const float *func) {
            auto builder2d = Dichotomy2D::create_builder(func, U, V);
            Array<StaticDichotomySampler<U>, V> conditional_v;
            for (int i = 0; i < builder2d.conditional_v.size(); ++i) {
                auto builder = builder2d.conditional_v[i];
                StaticDichotomySampler<U> static_distribution(builder);
                conditional_v[i] = static_distribution;
            }
            StaticDichotomySampler<V> marginal(builder2d.marginal);
            StaticDichotomy2D<U, V> ret(conditional_v, marginal);
            return ret;
        }

    } // luminous::sampling
} // luminous