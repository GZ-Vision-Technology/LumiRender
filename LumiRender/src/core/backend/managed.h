//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"

namespace luminous {
    template<typename THost, typename TDevice = std::remove_pointer_t<THost>>
    struct DeviceHandle {
    private:
        int _n_elements;
        THost _host;
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        DeviceHandle(THost host, const std::shared_ptr<Device> &device, int n = 1)
                : _n_elements(n), _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        THost get() {
            return _host;
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr();
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
                _device_buffer.upload(_host, _n_elements);
            } else {
                const THost *p = &_host;
                _device_buffer.upload(p, _n_elements);
            }
        }

        void synchronize_to_cpu() {
            if constexpr (std::is_pointer_v<THost>) {
                _device_buffer.download(_host, _n_elements);
            } else {
                _device_buffer.download(&_host, _n_elements);
            }
        }
    };
}