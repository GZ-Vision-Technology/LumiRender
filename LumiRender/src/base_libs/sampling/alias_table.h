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

        LM_ND_INLINE std::pair<std::vector<AliasEntry>, std::vector<float>>
        create_alias_table(BufferView<const float> values) {
            auto sum = std::reduce(values.cbegin(), values.cend(), 0.0);
            auto inv_sum = 1.0 / sum;
            std::vector<float> pdf(values.size());
            std::transform(
                    values.cbegin(), values.cend(), pdf.begin(),
                    [inv_sum](auto v) noexcept {
                        return static_cast<float>(v * inv_sum);
                    });

            auto ratio = static_cast<double>(values.size()) / sum;
            static thread_local std::vector<uint> over;
            static thread_local std::vector<uint> under;
            over.clear();
            under.clear();
            over.reserve(next_pow2(values.size()));
            under.reserve(next_pow2(values.size()));

            std::vector<AliasEntry> table(values.size());
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

            return std::make_pair(std::move(table), std::move(pdf));
        }

        struct AliasData {
        public:
            BufferView <AliasEntry> table{};
            BufferView<float> PDF{};
        };

        template<uint N>
        struct StaticAliasData {
        public:
            Array <AliasEntry, N> table;
            Array<float, N> PDF;
        };

        template<typename TData>
        class TAliasTable1D {
        private:
            TData _data;
        public:
            explicit TAliasTable1D(const TData &data) {

            }


            LM_ND_XPU int sample_discrete(float u, float *p = nullptr,
                                          float *u_remapped = nullptr) const;

            LM_ND_XPU float sample_continuous(float u, float *pdf = nullptr,
                                              int *ofs = nullptr) const;

            LM_ND_XPU size_t size() const;

            LM_ND_XPU float integral() const;

            LM_ND_XPU float func_at(uint32_t i) const;

            LM_ND_XPU float PMF(uint32_t i) const;
        };
    }
}