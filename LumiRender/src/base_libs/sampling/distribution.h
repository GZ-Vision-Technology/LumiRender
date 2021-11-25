//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/lstd/common.h"

namespace luminous {
    inline namespace sampling {
        using std::vector;
        using std::move;

        struct Distribution1DBuilder {
        public:
            std::vector<float> func;
            std::vector<float> CDF;
            float func_integral{};

            Distribution1DBuilder() = default;

            Distribution1DBuilder(std::vector<float> func, std::vector<float> CDF, float integral)
                    : func(move(func)), CDF(move(CDF)), func_integral(integral) {}
        };

        struct DistribData {
        public:
            using value_type = float;
            using const_value_type = const float;
        public:
            // todo change to indice mode, reduce memory usage
            BufferView <const_value_type> func{};
            BufferView <const_value_type> CDF{};
            float func_integral{};

            DistribData() = default;

            DistribData(BufferView <const_value_type> func,
                        BufferView <const_value_type> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}
        };

        template<int Size>
        struct CDistribData {
        private:
            static constexpr int size_in_bytes = Size * sizeof(float);
        public:
            Array<float, Size> func;
            Array<float, Size + 1> CDF;
            float func_integral{};

            CDistribData() = default;

            CDistribData(Array<float, Size> func,
                         Array<float, Size + 1> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}

            CDistribData(const float *f, const float *C, float integral) {
                init(f, C, integral);
            }

            void init(const float *f, const float *C, float integral) {
                std::memcpy(func.begin(), f, size_in_bytes);
                std::memcpy(CDF.begin(), C, size_in_bytes + sizeof(float));
                func_integral = integral;
            }
        };

        template<typename T = DistribData>
        class TDistribution {
        public:
            using data_type = T;
        private:
            data_type _data;
        public:
            TDistribution() = default;

            explicit TDistribution(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TDistribution(Args ...args) : TDistribution(T(std::forward<Args>(args)...)) {}

            LM_ND_XPU size_t size() const { return _data.func.size(); }

            LM_ND_XPU float sample_continuous(float u, float *pdf = nullptr, int *ofs = nullptr) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                size_t offset = find_interval((int) _data.CDF.size(), predicate);
                if (ofs) {
                    *ofs = offset;
                }
                float du = u - _data.CDF[offset];
                if ((_data.CDF[offset + 1] - _data.CDF[offset]) > 0) {
                    DCHECK_GT(_data.CDF[offset + 1], _data.CDF[offset]);
                    du /= (_data.CDF[offset + 1] - _data.CDF[offset]);
                }
                DCHECK(!is_nan(du));

                if (pdf) {
                    *pdf = (_data.func_integral > 0) ? _data.func[offset] / _data.func_integral : 0;
                }
                return (offset + du) / size();
            }

            LM_ND_XPU int sample_discrete(float u, float *p = nullptr, float *u_remapped = nullptr) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                int offset = find_interval(_data.CDF.size(), predicate);
                if (p) {
                    //todo
                    *p = PMF(offset);
                }
                if (u_remapped) {
                    *u_remapped = (u - _data.CDF[offset]) / (_data.CDF[offset + 1] - _data.CDF[offset]);
                    DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                }
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
                return {move(func), move(CDF), integral};
            }
        };

        using Distribution1D = TDistribution<DistribData>;

        template<int Size>
        using StaticDistribution1D = TDistribution<CDistribData<Size>>;

        struct Distribution2DBuilder {
            vector<Distribution1DBuilder> conditional_v;
            Distribution1DBuilder marginal;

            Distribution2DBuilder(vector<Distribution1DBuilder> conditional_v, Distribution1DBuilder marginal)
                    : conditional_v(move(conditional_v)), marginal(move(marginal)) {}
        };

        struct Distribution2DData {
        public:
            BufferView<const Distribution1D> conditional_v{};
            Distribution1D marginal{};

            Distribution2DData() = default;

            Distribution2DData(BufferView<const Distribution1D> conditional_v,
                               Distribution1D marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<int U, int V>
        struct CDistribution2DData {
        public:
            Array <StaticDistribution1D<U>, V> conditional_v{};
            StaticDistribution1D<V> marginal;

            CDistribution2DData() = default;

            CDistribution2DData(Array <StaticDistribution1D<U>, V> conditional_v,
                                StaticDistribution1D<V> marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<typename T>
        class TDistribution2D {
        public:
            using data_type = T;
        private:
            data_type _data;
        public:
            TDistribution2D() = default;

            explicit TDistribution2D(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TDistribution2D(Args ...args) : TDistribution2D(T(std::forward<Args>(args)...)) {}

            LM_ND_XPU float2 sample_continuous(float2 u, float *PDF, int2 *offset = nullptr) const {
                float PDFs[2];
                int2 uv;
                float d1 = _data.marginal.sample_continuous(u[1], &PDFs[1], &uv[1]);
                float d0 = _data.conditional_v[uv[1]].sample_continuous(u[0], &PDFs[0], &uv[0]);
                *PDF = PDFs[0] * PDFs[1];
                if (offset) {
                    *offset = uv;
                }
                return make_float2(d0, d1);
            }

            LM_ND_XPU float PDF(float2 p) const {
                size_t iu = clamp(size_t(p[0] * _data.conditional_v[0].size()), 0, _data.conditional_v[0].size() - 1);
                size_t iv = clamp(size_t(p[1] * _data.marginal.size()), 0, _data.marginal.size() - 1);
                return _data.conditional_v[iv].func_at(iu) / _data.marginal.integral();
            }

            static Distribution2DBuilder create_builder(const float *func, int nu, int nv) {
                vector<Distribution1DBuilder> conditional_v;
                conditional_v.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    vector<float> func_v;
                    func_v.insert(func_v.end(), &func[v * nu], &func[v * nu + nu]);
                    Distribution1DBuilder builder = Distribution1D::create_builder(func_v);
                    conditional_v.push_back(builder);
                }
                vector<float> marginal_func;
                marginal_func.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    marginal_func.push_back(conditional_v[v].func_integral);
                }
                Distribution1DBuilder marginal_builder = Distribution1D::create_builder(marginal_func);
                return {move(conditional_v), move(marginal_builder)};
            }
        };

        using Distribution2D = TDistribution2D<Distribution2DData>;

        template<int U, int V>
        using StaticDistribution2D = TDistribution2D<CDistribution2DData<U, V>>;

        template<int U, int V>
        LM_NODISCARD StaticDistribution2D<U, V> create_static_distrib2d(const float *func) {
            auto builder2d = Distribution2D::create_builder(func, U, V);
            Array <StaticDistribution1D<U>, V> conditional_v;
            for (int i = 0; i < builder2d.conditional_v.size(); ++i) {
                auto builder = builder2d.conditional_v[i];
                StaticDistribution1D<U> static_distribution(builder.func.data(),
                                                            builder.CDF.data(),
                                                            builder.func_integral);
                conditional_v[i] = static_distribution;
            }
            StaticDistribution1D<V> marginal(builder2d.marginal.func.data(),
                                             builder2d.marginal.CDF.data(),
                                             builder2d.marginal.func_integral);
            StaticDistribution2D<U, V> ret(conditional_v, marginal);
            return ret;
        }

    } // luminous::sampling
} // luminous