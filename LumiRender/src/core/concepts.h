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

    struct MovableNonCopyable {
        MovableNonCopyable() = default;
        MovableNonCopyable(MovableNonCopyable &&) = default;
        MovableNonCopyable& operator = (MovableNonCopyable &&) = default;

    private:
        MovableNonCopyable(const MovableNonCopyable &) = delete;
        MovableNonCopyable &operator=(const MovableNonCopyable &) = delete;
    };
}

