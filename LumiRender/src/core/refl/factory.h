//
// Created by Zero on 13/09/2021.
//


#pragma once

#include "base_libs/header.h"
#include <map>
#include <vector>
#include <string>
#include "core/logging.h"
#include "core/macro_map.h"

namespace luminous {

    inline namespace refl {
        class Object;

#define GET_PTR_VALUE(ptr, offset) ((reinterpret_cast<luminous::ptr_t*>(&((reinterpret_cast<std::byte *>(ptr))[offset])))[0])

#define SET_PTR_VALUE(ptr, offset, val) (reinterpret_cast<luminous::ptr_t*>(&((reinterpret_cast<std::byte*>(ptr))[offset])))[0] = ptr_t(val)

        template<typename T>
        LM_NODISCARD luminous::ptr_t get_ptr_value(T ptr, uint64_t offset) {
            static constexpr bool condition = std::is_pointer_v<T> || std::is_same_v<luminous::ptr_t, T>;
            static_assert(condition, "the type T must be ptr_t or pointer!");
            return GET_PTR_VALUE(ptr, offset);
        }

        template<typename T, typename U>
        INLINE void set_ptr_value(T ptr, uint64_t offset, U val) {
            static constexpr bool condition1 = std::is_pointer_v<T> || std::is_same_v<luminous::ptr_t, T>;
            static constexpr bool condition2 = std::is_pointer_v<U> || std::is_same_v<luminous::ptr_t, U>;
            static_assert(condition2 && condition1, "the type T and U must be ptr_t or pointer!");
            SET_PTR_VALUE(ptr, offset, val);
        }

#undef SET_PTR_VALUE
#undef GET_PTR_VALUE

        // class Object : public BaseBinder<> {
        // public:
        //     REFL_CLASS(Object)
        // };
    }
}