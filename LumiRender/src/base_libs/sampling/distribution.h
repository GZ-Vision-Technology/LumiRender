//
// Created by Zero on 13/02/2022.
//


#pragma once

#include "alias_table.h"
#include "dichotomy.h"

namespace luminous {
    inline namespace sampling {

#if USE_ALIAS_TABLE
        using Distribution1D = TAliasTable<AliasData>;

        template<int Size>
        using StaticDistribution1D = TAliasTable<StaticAliasData<Size>>;
#else
        using Distribution1D = TDichotomySampler<DichotomyData>;

        template<int Size>
        using StaticDistribution1D = TDichotomySampler<StaticDichotomyData<Size>>;
#endif

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
        struct StaticDistribution2DData {
        public:
            Array<StaticDistribution1D<U>, V> conditional_v{};
            StaticDistribution1D<V> marginal;

            StaticDistribution2DData() = default;

            StaticDistribution2DData(Array<StaticDistribution1D<U>, V> conditional_v,
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

//            static Builder create_builder(const float *func, int nu, int nv) {
//                vector<Builder> conditional_v;
//                conditional_v.reserve(nv);
//                for (int v = 0; v < nv; ++v) {
//                    vector<float> func_v;
//                    func_v.insert(func_v.end(), &func[v * nu], &func[v * nu + nu]);
//                    Builder builder = Distribution1D::create_builder(func_v);
//                    conditional_v.push_back(builder);
//                }
//                vector<float> marginal_func;
//                marginal_func.reserve(nv);
//                for (int v = 0; v < nv; ++v) {
//                    marginal_func.push_back(conditional_v[v].func_integral);
//                }
//                Builder marginal_builder = Distribution1D::create_builder(marginal_func);
//                return {move(conditional_v), move(marginal_builder)};
//            }
        };

        using Distribution2D = TDistribution2D<Dichotomy2DData>;

        template<int U, int V>
        using StaticDistribution2D = TDistribution2D<StaticDichotomy2DData<U, V>>;
    }
}