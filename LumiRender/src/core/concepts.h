//
// Created by Zero on 2020/9/1.
//
#pragma once

#include "base_libs/header.h"

namespace luminous {

    struct Noncopyable {
        Noncopyable() = default;

        Noncopyable(const Noncopyable &) = delete;

        Noncopyable &operator=(const Noncopyable &) = delete;
    };

    template<typename T>
    class Creator {
    public:
        CPU_ONLY(
                template<typename ...Args>
                static T create(Args &&...args) {
                    return T(std::forward<Args>(args)...);
                }
        )

        CPU_ONLY(
                template<typename ...Args>
                static T *create_ptr(Args &&...args) {
                    return new T(std::forward<Args>(args)...);
                }
        )

    CHECK_PLACEHOLDER
    };

}

