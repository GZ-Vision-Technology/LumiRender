//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename T>
    struct Managed : Noncopyable {
    public:
        using value_type = T;
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<T>>, "T can not be the secondary pointer!");
        size_t _n_elements{0};
        std::vector<T> _host{};
        Buffer <T> _device_buffer{nullptr};

    public:
        Managed() = default;

        Managed(T *host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<T>(n);
        }

        Managed(Managed<T> &&other) {
            _host = move(other._host);
            _device_buffer = move(other._device_buffer);
            _n_elements = other._n_elements;
        }

        size_t size_in_bytes() const {
            return _device_buffer.size_in_bytes();
        }

        size_t size() const {
            return _n_elements;
        }

        void reset(T *host, const SP <Device> &device, int n = 1) {
            _n_elements = n;
            _device_buffer = device->allocate_buffer<T>(_n_elements);
            _host.reserve(_n_elements);
            _host.resize(_n_elements);
            std::memcpy(_host.data(), host, sizeof(T) * _n_elements);
        }

        void reset(T *host, int n = 1) {
            _n_elements = n;
            _host.reserve(_n_elements);
            _host.resize(_n_elements);
            std::memcpy(_host.data(), host, sizeof(T) * _n_elements);
        }

        void reset(std::vector<T> v) {
            _n_elements = v.size();
            _host = std::move(v);
        }

        void reset(std::vector<T> v, const SP <Device> &device) {
            _n_elements = v.size();
            _host = std::move(v);
            _device_buffer = device->allocate_buffer<T>(_n_elements);
        }

        void reset(size_t n) {
            _n_elements = n;
            _host.resize(n);
            std::memset(_host.data(), 0, sizeof(T) * _n_elements);
        }

        void reset(const SP <Device> &device, size_t n) {
            _n_elements = n;
            _host.resize(n);
            _device_buffer = device->allocate_buffer<T>(_n_elements);
            std::memset(_host.data(), 0, sizeof(T) * _n_elements);
        }

        template<typename... Args>
        void reset_all(Args &&...args) {
            reset(std::forward<Args>(args)...);
            synchronize_to_gpu();
        }

        void allocate_device(const SP <Device> device) {
            _device_buffer = device->allocate_buffer<T>(_n_elements);
        }

        BufferView <T> host_buffer_view(size_t offset = 0, size_t count = 0) const {
            count = fix_count(offset, count, size());
            return BufferView<T>(((T *) _host.data()) + offset, count);
        }

        BufferView <T> device_buffer_view(size_t offset = 0, size_t count = 0) const {
            return _device_buffer.view(offset, count);
        }

        const Buffer <T> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            _device_buffer.clear();
            _host.clear();
        }

        void clear_host() {
            _host.clear();
        }

        void clear_device() {
            _device_buffer.clear();
        }

        const std::vector<T> &c_vector() const {
            return _host;
        }

        std::vector<T> &vector() {
            return _host;
        }

        T *get() {
            return reinterpret_cast<T *>(_host.data());
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        T *device_data() const {
            return _device_buffer.data();
        }

        template<typename Index>
        T &operator[](Index i) {
            DCHECK(i < _device_buffer.size());
            return _host[i];
        }

        template<typename Index>
        const T &operator[](Index i) const {
            DCHECK(i < _device_buffer.size());
            return _host[i];
        }

        T *operator->() {
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