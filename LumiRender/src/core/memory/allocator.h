//
// Created by Zero on 17/09/2021.
//


#pragma once

#include <vector>

namespace luminous {
    inline namespace core {
        template<typename T>
        class Allocator : public std::allocator<T> {
        public:
            using value_type = T;
            using pointer = T *;
            using const_pointer = const T *;
            using void_pointer = void *;
            using const_void_pointer = const void *;
            using size_type = size_t;
            using difference_type = std::ptrdiff_t;

        public:
            Allocator() = default;

            ~Allocator() = default;
        };
    }
}