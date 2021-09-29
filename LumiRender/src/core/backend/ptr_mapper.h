//
// Created by Zero on 17/09/2021.
//


#pragma once

#include "base_libs/math/interval.h"
#include <vector>
#include <queue>
#include <map>

namespace luminous {
    inline namespace core {


        class PtrMapper {
        private:
            static PtrMapper *_instance;

            std::map<ptr_t, std::pair<PtrInterval, PtrInterval>> _map;

            // for fast mapping
            std::map<ptr_t, ptr_t> _device_to_host;

            PtrMapper() = default;

            PtrMapper(const PtrMapper &) = default;

            PtrMapper &operator=(const PtrMapper &) = default;

        public:
            static PtrMapper *instance();

            void add_pair(PtrInterval host, PtrInterval device);

            void add_reverse_mapping(ptr_t device_ptr, ptr_t host_ptr);

            ptr_t get_host_ptr(ptr_t device_ptr) const;

            template<typename T>
            LM_NODISCARD ptr_t get_device_ptr(T host_ptr) const {
                return get_device_ptr(ptr_t(host_ptr));
            }

            LM_NODISCARD ptr_t get_device_ptr(ptr_t  host_ptr) const;
        };
    }
}