//
// Created by Zero on 2021/3/9.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "kernel.h"

namespace luminous {
    class Module {
    public:
        class Impl {
        public:
            LM_NODISCARD virtual SP <KernelOld> get_kernel(const std::string &name) = 0;

            LM_NODISCARD virtual std::pair<ptr_t, size_t> get_global_var(const std::string &name) = 0;

            LM_NODISCARD virtual uint64_t get_kernel_handle(const std::string &name) = 0;

            virtual void upload_data_to_global_var(const std::string &name, const void *data, size_t size) = 0;
        };

        LM_NODISCARD SP <KernelOld> get_kernel(const std::string &name) {
            return _impl->get_kernel(name);
        }

        LM_NODISCARD uint64_t get_kernel_handle(const std::string &name) {
            return _impl->get_kernel_handle(name);
        }

        LM_NODISCARD std::pair<ptr_t, size_t> get_global_var(const std::string &name) {
            return _impl->get_global_var(name);
        }


        template<typename T>
        void upload_data_to_global_var(const std::string &name, const T &data) {
            _impl->upload_data_to_global_var(name, data, sizeof(T));
        }

        explicit Module(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    private:
        std::unique_ptr<Impl> _impl;
    };
}