//
// Created by Zero on 2020/9/1.
//
#pragma once

#include "header.h"

LUMINOUS_BEGIN

inline namespace utility {

    class IObject {

    };

    struct Noncopyable {
        Noncopyable() = default;

        Noncopyable(const Noncopyable &) = delete;

        Noncopyable &operator=(const Noncopyable &) = delete;
    };



};

LUMINOUS_END

