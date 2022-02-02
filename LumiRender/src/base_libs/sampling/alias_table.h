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
        create_alias_table(BufferView<const float> values) {
            auto sum = std::reduce(values.cbegin(), values.cend(), 0.0);
            auto inv_sum = 1.0 / sum;
            vector<float> pmf(values.size());
            std::transform(
                    values.cbegin(), values.cend(), pmf.begin(),
                    [inv_sum](auto v) noexcept {
                        return static_cast<float>(v * inv_sum);
                    });

            auto ratio = static_cast<double>(values.size()) / sum;
            static thread_local vector<uint> over;
            static thread_local vector<uint> under;
            over.clear();
            under.clear();
            over.reserve(next_pow2(values.size()));
            under.reserve(next_pow2(values.size()));

            vector<AliasEntry> table(values.size());
            for (auto i = 0u; i < values.size(); i++) {
                auto p = static_cast<float>(values[i] * ratio);
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
            BufferView <AliasEntry> table{};
            BufferView<float> PMF{};
        };

        template<uint N>
        struct StaticAliasData {
        public:
            Array <AliasEntry, N> table;
            Array<float, N> PMF;
        };

        template<typename TData>
        class TAliasTable {
        private:
            TData _data;
        public:
            explicit TAliasTable(const TData &data)
                    : _data(data) {}

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
        };
    }
}