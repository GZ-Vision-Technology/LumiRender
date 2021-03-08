//
// Created by Zero on 2021/3/9.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "kernel.h"

namespace luminous {
    class Module {
    public:
        class Impl {
        public:
            virtual SP<Kernel> get_kernel(const std::string &name) = 0;
        };

        SP<Kernel> get_kernel(const std::string &name) {
            return impl->get_kernel(name);
        }

        explicit Module(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
    private:
        std::unique_ptr<Impl> impl;
    };
}