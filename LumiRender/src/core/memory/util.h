//
// Created by Zero on 07/01/2022.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace core {
        template<typename T = void>
        LM_NODISCARD T *aligned_alloc(size_t alignment, size_t size_in_byte) noexcept {
            return reinterpret_cast<T *>(_aligned_malloc(size_in_byte, alignment));
        }

        template<typename T = std::byte>
        LM_NODISCARD T *aligned_alloc(size_t num) noexcept {
            return aligned_alloc<T>(alignof(T), num * sizeof(T));
        }

        template<typename T = void>
        void aligned_free(T *p) noexcept {
            _aligned_free(p);
        }

        template<typename T, typename... Args>
        constexpr T *construct_at(T *p, Args &&...args) {
            return ::new(const_cast<void *>(static_cast<const volatile void *>(p)))
                    T(std::forward<Args>(args)...);
        }

        template<typename T = std::byte>
        LM_NODISCARD T *new_array(size_t num) noexcept {
            return new T[num];
        }

        template<typename T, typename... Args>
        LM_NODISCARD T *create(Args &&...args) {
            return construct_at(new T(), std::forward<Args>(args)...);
        }
    }
}