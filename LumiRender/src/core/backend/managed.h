//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename THost, typename TDevice = THost>
    struct Managed : Noncopyable {
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        size_t _n_elements{0};
        std::vector<THost> _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed(THost *host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed() {}

        void reset(THost *host, const SP <Device> &device, int n = 1) {
            _n_elements = n;
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            _host.reserve(_n_elements);
            _host.resize(_n_elements);
            std::memcpy(_host.data(), host, sizeof(THost) * _n_elements);
        }

        void reset(std::vector<THost> v, const SP <Device> &device) {
            _n_elements = v.size();
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            _host = std::move(v);
        }

        void reset(const SP <Device> &device, size_t n) {
            _n_elements = n;
            _host.resize(n);
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            std::memset(_host.data(), 0, sizeof(THost) * _n_elements);
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            _device_buffer.clear();
            _host.reset(nullptr);
        }

        const std::vector<THost> &c_vector() const {
            return _host;
        }

        std::vector<THost> &vector() {
            return _host;
        }

        THost *get() {
            return _host.data();
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        TDevice *device_data() const {
            return _device_buffer.data();
        }

        THost operator[](uint i) {
            assert(i < _device_buffer.size());
            return _host[i];
        }

        auto operator->() {
            return _host.data();
        }

        void synchronize_to_gpu() {
            _device_buffer.upload(_host.data());
        }

        void synchronize_to_cpu() {
            _device_buffer.download(_host.data());
        }
    };
}