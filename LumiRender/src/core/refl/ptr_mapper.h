//
// Created by Zero on 17/09/2021.
//


#pragma once

#include "base_libs/math/interval.h"
#include <vector>
#include <queue>
#include <map>

namespace luminous {
    inline namespace refl {


        class PtrMapper {
        private:
            static PtrMapper *_instance;

            std::map<ptr_t, std::pair<PtrInterval, PtrInterval>> _map;

            PtrMapper() = default;

            PtrMapper(const PtrMapper &) = default;

            PtrMapper &operator=(const PtrMapper &) = default;

        public:
            static PtrMapper *instance();

            void add_pair(PtrInterval host, PtrInterval device);

            LM_NODISCARD ptr_t get_device_ptr(ptr_t  host_ptr) const;
        };
    }
}