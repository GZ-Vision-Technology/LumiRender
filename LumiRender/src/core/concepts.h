//
// Created by Zero on 2020/9/1.
//
#pragma once

#include "graphics/header.h"


namespace luminous {

    class IObject {

    };

    struct Noncopyable {
        Noncopyable() = default;

        Noncopyable(const Noncopyable &) = delete;

        Noncopyable &operator=(const Noncopyable &) = delete;
    };
}

