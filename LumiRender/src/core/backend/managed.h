//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename THost, typename TDevice = std::remove_pointer_t<THost>>
    struct Managed : Noncopyable {
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        THost _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed(THost host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed() {}

        void reset(THost host, const SP <Device> &device, int n = 1) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
            _host = host;
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        THost get() {
            return _host;
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        auto device_data() const {
            return _device_buffer.data();
        }

        auto operator[](uint i) {
            static_assert(std::is_pointer_v<THost>, "subscript only operate pointer!");
            assert(i < _device_buffer.size());
            return _host[i];
        }

        auto operator->() {
            if constexpr (std::is_pointer_v<THost>) {
                return _host;
            } else {
                return &_host;
            }
        }

        void synchronize_to_gpu() {
            if constexpr (std::is_pointer_v<THost>) {
                _device_buffer.upload(_host);
            } else {
                _device_buffer.upload(&_host);
            }
        }

        void synchronize_to_cpu() {
            if constexpr (std::is_pointer_v<THost>) {
                _device_buffer.download(_host);
            } else {
                _device_buffer.download(&_host);
            }
        }
    };
}