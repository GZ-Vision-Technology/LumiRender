//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"
#include "core/memory/allocator.h"

namespace luminous {
    template<typename T, typename U = const T, typename AlTy = core::Allocator<T>>
    struct Managed : public Noncopyable, public std::vector<T, AlTy> {
    public:
        using BaseClass = std::vector<T, AlTy>;
        using THost = T;
        using TDevice = U;
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        Buffer<TDevice> _device_buffer{nullptr};
        Device *_device{};
    public:
        Managed() = default;

        explicit Managed(Device *device) : _device(device) {}

        Managed(Managed &&other) noexcept
                : BaseClass(std::move(other)),
                  _device_buffer(std::move(other._device_buffer)) {}

        NDSC size_t size_in_bytes() const {
            return BaseClass::size() * sizeof(T);
        }

        void set_device(Device *device) { _device = device; }

        void reset(THost *host, int n = 1) {
            BaseClass::reserve(n);
            BaseClass::resize(n);
            std::memcpy(BaseClass::data(), host, sizeof(THost) * n);
            void *ptr = _device->is_cpu() ? BaseClass::data() : nullptr;
            _device_buffer = _device->create_buffer<TDevice>(n, ptr);
        }

        void reset(const std::vector<THost> &v) {
            reset(v.data(), v.size());
        }

        void reset(size_t n) {
            BaseClass::resize(n);
            void *ptr = _device->is_cpu() ? BaseClass::data() : nullptr;
            _device_buffer = _device->create_buffer<TDevice>(n, ptr);
            std::memset(BaseClass::data(), 0, sizeof(THost) * n);
        }

        void append(const std::vector<THost> &v) {
            BaseClass::insert(BaseClass::cend(), v.cbegin(), v.cend());
        }

        void allocate_device(size_t size = 0) {
            size = size == 0 ? BaseClass::size() : size;
            if (size == 0) {
                return;
            }
            void *ptr = _device->is_cpu() ? BaseClass::data() : nullptr;
            _device_buffer = _device->create_buffer<TDevice>(size, ptr);
        }

        NDSC BufferView<THost> obtain_accessible_buffer_view(size_t offset = 0, size_t count = -1) {
            if (_device->is_cpu()) {
                return host_buffer_view(offset, count);
            } else {
                synchronize_to_host();
                return device_buffer_view(offset, count);
            }
        }

        NDSC BufferView<const THost> obtain_const_accessible_buffer_view(size_t offset = 0, size_t count = -1) {
            if (_device->is_cpu()) {
                return static_cast<BufferView<const Vector<float, 4>> &&>(host_buffer_view(offset, count));
            } else {
                synchronize_to_host();
                return device_buffer_view(offset, count);
            }
        }

        NDSC BufferView<const THost> const_host_buffer_view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, BaseClass::size());
            return BufferView<const THost>(((const THost *) BaseClass::data()) + offset, count);
        }

        NDSC BufferView<THost> host_buffer_view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, BaseClass::size());
            return BufferView<THost>(((THost *) BaseClass::data()) + offset, count);
        }

        NDSC BufferView<TDevice> device_buffer_view(size_t offset = 0, size_t count = -1) const {
            return _device_buffer.view(offset, count);
        }

        NDSC BufferView<const TDevice> const_device_buffer_view(size_t offset = 0, size_t count = -1) const {
            return _device_buffer.view(offset, count);
        }

        const Buffer<TDevice> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            clear_device();
            clear_host();
        }

        void clear_host() {
            BaseClass::clear();
        }

        void clear_device() {
            _device_buffer.clear();
        }

        template<typename pointer_type = void *>
        pointer_type device_ptr() const {
            return _device_buffer.template ptr<pointer_type>();
        }

        TDevice *device_data() const {
            return _device_buffer.data();
        }

        const THost *operator->() const {
            return BaseClass::data();
        }

        THost *operator->() {
            return BaseClass::data();
        }

        void synchronize_to_device() {
            if (BaseClass::size() == 0) {
                return;
            }
            _device_buffer.upload(BaseClass::data());
        }

        void synchronize_to_host() {
            if (BaseClass::size() == 0) {
                BaseClass::resize(_device_buffer.size());
            }
            _device_buffer.download(BaseClass::data());
        }
    };
}