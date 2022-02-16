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
    inline namespace sampling {
        struct AliasEntry {
            float prob;
            uint alias;
        };

        using std::vector;
        using std::pair;

        struct AliasTableBuilder {
            vector<AliasEntry> table;
            vector<float> func;
            float func_integral{};
        };

        LM_ND_INLINE auto create_alias_table(vector<float> weights) {
            auto sum = std::reduce(weights.cbegin(), weights.cend(), 0.0);
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

            return std::tuple{std::move(table), std::move(weights), float(sum / weights.size())};
        }

        struct AliasData {
        public:
            BufferView<const AliasEntry> table{};
            BufferView<const float> func{};
            float func_integral{};
        public:
            AliasData() = default;

            AliasData(BufferView<const AliasEntry> t, BufferView<const float> p, float integral)
                    : table(t), func(p), func_integral(integral) {}
        };

        template<uint N>
        struct StaticAliasData {
        public:
            using Builder = AliasTableBuilder;
        public:
            Array <AliasEntry, N> table;
            Array<float, N> func;
            float func_integral{};
        public:
            StaticAliasData() = default;

            StaticAliasData(Array <AliasEntry, N> t, Array<float, N> p, float integral)
                    : table(t), func(p), func_integral(integral) {}

            StaticAliasData(const AliasEntry *alias_entry, const float *p, float integral) {
                init(alias_entry, p, integral);
            }

            explicit StaticAliasData(const Builder &builder) {
                init(builder.table.data(), builder.func.data(), builder.func_integral);
            }

            void init(const AliasEntry *alias_entry, const float *p, float func_int) {
                std::memcpy(table.begin(), alias_entry, N * sizeof(AliasEntry));
                std::memcpy(func.begin(), p, N * sizeof(float));
                this->func_integral = func_int;
            }
        };

        template<typename TData>
        class TAliasTable {
        public:
            using data_type = TData;
            using Builder = AliasTableBuilder;
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
                offset = select(u < alias_entry.prob, offset, alias_entry.alias);
                *p = PMF(offset);
                *u_remapped = select(u < alias_entry.prob,
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
                *ofs = select(u < alias_entry.prob, *ofs, alias_entry.alias);
                *p = PDF(*ofs);
                float u_remapped = select(u < alias_entry.prob,
                                          std::min<float>(u / alias_entry.prob, OneMinusEpsilon),
                                          std::min<float>((1 - u) / (1 - alias_entry.prob), OneMinusEpsilon));
                return (*ofs + u_remapped) / size();
            }

            LM_ND_XPU float integral() const { return _data.func_integral; }

            LM_ND_XPU float func_at(uint32_t i) const { return _data.func[i]; }

            LM_ND_XPU size_t size() const { return _data.func.size(); }

            LM_ND_XPU float PDF(uint32_t i) const {
                DCHECK(i < size());
                float f = func_at(i);
                return integral() > 0 ? (func_at(i) / integral()) : 0;
            }

            LM_ND_XPU float PMF(uint32_t i) const {
                DCHECK(i < size());
                float f = func_at(i);
                return integral() > 0 ? (func_at(i) / (integral() * size())) : 0;
            }

            static AliasTableBuilder create_builder(std::vector<float> weights) {
                auto[table, func, integral] = create_alias_table(std::move(weights));
                return {table, func, integral};
            }
        };
    }
}