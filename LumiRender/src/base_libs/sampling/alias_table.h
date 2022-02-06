//
// Created by Zero on 29/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/lstd/common.h"
#include <numeric>
#include <vector>

namespace luminous {
    inline namespace render {
        struct AliasEntry {
            float prob;
            uint alias;
        };

        using std::vector;
        using std::pair;

        struct AliasTableBuilder {
            vector<AliasEntry> table;
            vector<float> PMF;
        };

        LM_ND_INLINE pair<vector<AliasEntry>, vector<float>>
        create_alias_table(vector<float> weights) {
            auto sum = std::reduce(weights.cbegin(), weights.cend(), 0.0);
            auto inv_sum = 1.0 / sum;
            vector<float> pmf(weights.size());
            std::transform(
                    weights.cbegin(), weights.cend(), pmf.begin(),
                    [inv_sum](auto v) noexcept {
                        return static_cast<float>(v * inv_sum);
                    });

            auto ratio = static_cast<double>(weights.size()) / sum;
            static thread_local vector<uint> over;
            static thread_local vector<uint> under;
            over.clear();
            under.clear();
            over.reserve(next_pow2(weights.size()));
            under.reserve(next_pow2(weights.size()));

            vector<AliasEntry> table(weights.size());
            for (auto i = 0u; i < weights.size(); i++) {
                auto p = static_cast<float>(weights[i] * ratio);
                table[i] = {p, i};
                (p > 1.0f ? over : under).emplace_back(i);
            }

            while (!over.empty() && !under.empty()) {
                auto o = over.back();
                auto u = under.back();
                over.pop_back();
                under.pop_back();
                table[o].prob -= 1.0f - table[u].prob;
                table[u].alias = o;
                if (table[o].prob > 1.0f) {
                    over.push_back(o);
                } else if (table[o].prob < 1.0f) {
                    under.push_back(o);
                }
            }
            for (auto i : over) { table[i] = {1.0f, i}; }
            for (auto i : under) { table[i] = {1.0f, i}; }

            return std::make_pair(std::move(table), std::move(pmf));
        }

        struct AliasData {
        public:
            BufferView<const AliasEntry> table{};
            BufferView<const float> PMF{};
        public:
            AliasData() = default;

            AliasData(BufferView<const AliasEntry> t, BufferView<const float> p)
                    : table(t), PMF(p) {}
        };

        template<uint N>
        struct StaticAliasData {
        public:
            Array <AliasEntry, N> table;
            Array<float, N> PMF;
        public:
            StaticAliasData() = default;

            StaticAliasData(Array <AliasEntry, N> t, Array<float, N> p)
                    : table(t), PMF(p) {}

            StaticAliasData(const AliasEntry *alias_entry, const float *p) {
                init(alias_entry, p);
            }

            void init(const AliasEntry *alias_entry, const float *p) {
                std::memcpy(table.begin(), alias_entry, N * sizeof(AliasEntry));
                std::memcpy(PMF.begin(), p, N * sizeof(float));
            }
        };

        template<typename TData>
        class TAliasTable {
        public:
            using data_type = TData;
        private:
            data_type _data;
        public:
            LM_XPU TAliasTable() = default;

            LM_XPU explicit TAliasTable(const TData &data)
                    : _data(data) {}

            template<typename ...Args>
            explicit TAliasTable(Args ...args) : TAliasTable(data_type(std::forward<Args>(args)...)) {}

            LM_ND_XPU int sample_discrete(float u, float *p,
                                          float *u_remapped) const {
                u = u * size();
                int offset = std::min<int>(int(u), size() - 1);
                u = std::min<float>(u - offset, OneMinusEpsilon);
                AliasEntry alias_entry = _data.table[offset];
                offset = select(alias_entry.prob < u, offset, alias_entry.alias);
                *p = PMF(offset);
                *u_remapped = select(alias_entry.prob < u,
                                     std::min<float>(u / alias_entry.prob, OneMinusEpsilon),
                                     std::min<float>((1 - u) / (1 - alias_entry.prob), OneMinusEpsilon));
                DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                return offset;
            }

            LM_ND_XPU float sample_continuous(float u, float *p,
                                              int *ofs) const {
                u = u * size();
                *ofs = std::min<int>(int(u), size() - 1);
                u = std::min<float>(u - *ofs, OneMinusEpsilon);
                AliasEntry alias_entry = _data.table[*ofs];
                *ofs = select(alias_entry.prob < u, *ofs, alias_entry.alias);
                *p = PMF(*ofs);
                float u_remapped = select(alias_entry.prob < u,
                                          std::min<float>(u / alias_entry.prob, OneMinusEpsilon),
                                          std::min<float>((1 - u) / (1 - alias_entry.prob), OneMinusEpsilon));
                return (*ofs + u_remapped) / size();
            }

            LM_ND_XPU size_t size() const { return _data.PMF.size(); }

            LM_ND_XPU float PMF(uint32_t i) const {
                DCHECK(i < size());
                return _data.PMF[i];
            }

            static AliasTableBuilder create_builder(std::vector<float> weights) {
                auto[table, PMF] = create_alias_table(std::move(weights));
                return {table, PMF};
            }
        };

        using AliasTable = TAliasTable<AliasData>;

        template<uint N>
        using StaticAliasTable = TAliasTable<StaticAliasData<N>>;

        struct AliasTable2DBuilder {
            vector<AliasTableBuilder> conditional_v;
            AliasTableBuilder marginal;
        };

        struct AliasData2D {
            BufferView<const AliasTable> conditional_v{};
            AliasTable marginal{};

            AliasData2D() = default;

            AliasData2D(BufferView<const AliasTable> conditional_v,
                        AliasTable marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<int U, int V>
        struct StaticAliasData2D {
        public:
            Array <StaticAliasTable<U>, V> conditional_v{};
            StaticAliasTable<V> marginal;

            StaticAliasData2D() = default;

            StaticAliasData2D(Array <StaticAliasTable<U>, V> conditional_v,
                              StaticAliasTable<V> marginal)
                    : conditional_v(conditional_v),
                      marginal(marginal) {}
        };

        template<typename TData>
        class TAliasTable2D {
        public:
            using data_type = TData;
        private:
            data_type _data;

        public:
            TAliasTable2D() = default;

            explicit TAliasTable2D(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TAliasTable2D(Args ...args) : TAliasTable2D(data_type(std::forward<Args>(args)...)) {}

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

            static AliasTable2DBuilder create_builder(const float *func, int nu, int nv) {
                vector<AliasTableBuilder> conditional_v;
                conditional_v.reserve(nv);
                vector<float> integrals;
                integrals.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    vector<float> func_v;
                    func_v.insert(func_v.end(), &func[v * nu], &func[v * nu + nu]);
                    AliasTableBuilder builder = AliasTable::create_builder(func_v);
                    integrals.push_back(std::reduce(func_v.cbegin(), func_v.cend(), 0.0));
                    conditional_v.push_back(builder);
                }
                vector<float> marginal_func;
                marginal_func.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    marginal_func.push_back(integrals[v]);
                }
                AliasTableBuilder marginal_builder = AliasTable::create_builder(marginal_func);
                return {move(conditional_v), marginal_builder};
            }
        };

        using AliasTable2D = TAliasTable2D<AliasData2D>;

        template<int U, int V>
        using StaticAliasTable2D = TAliasTable2D<StaticAliasData2D<U, V>>;

        template<int U, int V>
        LM_NODISCARD StaticAliasTable2D<U, V> create_static_alias_table2d(const float *func) {
            auto builder2d = AliasTable2D::create_builder(func, U, V);
            Array<StaticAliasTable<U>, V> conditional_v;
            for (int i = 0; i < builder2d.conditional_v.size(); ++i) {
                auto builder = builder2d.conditional_v[i];
                StaticAliasTable<U> static_alias_table(builder.table.data(),
                                                       builder.PMF.data());
                conditional_v[i] = static_alias_table;
            }
            StaticAliasTable<V> marginal(builder2d.marginal.table.data(),
                                         builder2d.marginal.PMF.data());
            StaticAliasTable2D<U, V> ret(conditional_v, marginal);
            return ret;
        }
    }
}