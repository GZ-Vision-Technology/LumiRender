//
// Created by Zero on 29/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/lstd/common.h"

namespace luminous {
    inline namespace render {
        class AliasTable1D {
        private:

        public:
            AliasTable1D(const BufferView<const float> &bfv) {
                init(bfv);
            }

            void init(const BufferView<const float> bfv);

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