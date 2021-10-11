//
// Created by Zero on 29/09/2021.
//


#pragma once

#include "managed.h"
#include "core/memory/arena.h"
#include "core/refl/reflection.h"
#include "core/refl/factory.h"
#include "base_libs/lstd/variant.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace core {

        /**
         * synchronize between host and device
         * use for instance of class T, that T has pointer member
         * @tparam T
         */
        template<typename T>
        class Synchronizer : public Managed<T, std::byte> {
        public:
            using element_type = T;
            using BaseClass = Managed<T, std::byte>;
        private:
            MemoryBlock _memory_block{};
            size_t _size_in_bytes{0};
            std::vector<size_t> _address;
        public:
            explicit Synchronizer(Device *device)
                    : BaseClass(device) {}

            void init(size_t n_element = 1, size_t size = lstd::Sizer<T>::max_size) {
                _size_in_bytes = size * n_element;
                USE_BLOCK(&_memory_block);
                _memory_block.allocate(_size_in_bytes);
                BaseClass::reserve(n_element);
                BaseClass::allocate_device(_size_in_bytes);
            }

            LM_NODISCARD ptr_t to_host_ptr(ptr_t ptr) const {
                DCHECK(is_device_ptr(ptr));
                std::ptrdiff_t ptrdiff = ptr - BaseClass::_device_buffer.template ptr<ptr_t>();
                return reinterpret_cast<ptr_t>(BaseClass::data()) + ptrdiff;
            }

            LM_NODISCARD ptr_t to_device_ptr(ptr_t ptr) const {
                DCHECK(is_host_ptr(ptr));
                std::ptrdiff_t ptrdiff = ptr - reinterpret_cast<ptr_t>(BaseClass::data());
                return BaseClass::_device_buffer.template ptr<ptr_t>() + ptrdiff;
            }

            LM_NODISCARD ptr_t host_ptr_head() const {
                return reinterpret_cast<ptr_t>(BaseClass::data());
            }

            LM_NODISCARD ptr_t device_ptr_head() const {
                return BaseClass::_device_buffer.template ptr<ptr_t>();
            }

            template<typename U>
            void add_element(const U &config) {
                USE_BLOCK(&_memory_block);
                std::pair<element_type, std::vector<size_t>> elm = element_type::create(config);
                lstd::append(_address, elm.second);
                BaseClass::push_back(elm.first);
                refl::for_each_all_registered_member<T>([&](size_t offset, const char *name, auto __) {
                    auto &element = BaseClass::back();
                    _address.push_back(ptr_t(&element) + offset);
                });
            }

            LM_NODISCARD const element_type *operator->() const {
                return BaseClass::data();
            }

            LM_NODISCARD element_type *operator->() {
                auto &ret = BaseClass::at(0);
                remapping_ptr_to_host();
                return &ret;
            }

            template<typename Index>
            LM_NODISCARD const element_type &at(Index i) const {
                return BaseClass::at(i);
            }

            template<typename Index>
            LM_NODISCARD element_type &at(Index i) {
                auto &ret = BaseClass::at(i);
                remapping_ptr_to_host();
                return ret;
            }

            template<typename Index>
            LM_NODISCARD element_type at(Index i) {
                auto ret = BaseClass::at(i);
                remapping_ptr_to_host();
                return ret;
            }

            template<typename Index>
            LM_NODISCARD element_type& operator[](Index i) {
                auto &ret = BaseClass::at(i);
                remapping_ptr_to_host();
                return ret;
            }

            LM_NODISCARD const element_type *device_ptr() const {
                return BaseClass::_device_buffer.template ptr<const element_type *>();
            }

            LM_NODISCARD element_type *device_ptr() {
                return BaseClass::_device_buffer.template ptr<element_type *>();
            }

            LM_NODISCARD BufferView<element_type> device_buffer_view(size_t offset = 0, size_t count = -1) {
                count = fix_count(offset, count, BaseClass::size());
                return BufferView<element_type>(BaseClass::_device_buffer.template ptr<element_type *>() + offset,
                                                count);
            }

            LM_NODISCARD BufferView<const element_type>
            const_device_buffer_view(size_t offset = 0, size_t count = -1) const {
                count = fix_count(offset, count, BaseClass::size());
                return BufferView<const element_type>(BaseClass::_device_buffer.template ptr<element_type *>() + offset,
                                                      count);
            }

            LM_NODISCARD bool is_device_ptr(ptr_t ptr) const {
                return BaseClass::device_interval().contains(ptr);
            }

            LM_NODISCARD bool is_host_ptr(ptr_t ptr) const {
                return _memory_block.interval_allocated().contains(ptr);
            }

            void remapping_ptr_to_host(element_type &elm) const {
                for_each_all_registered_member<T>([&](size_t offset, const char *name, auto __) {
                    ptr_t ptr = get_ptr_value(&elm, offset);
                    if (is_host_ptr(ptr)) {
                        return;
                    }
                    auto host_ptr = to_host_ptr(ptr);
                    set_ptr_value(ptr_t(0), ptr_t(&elm) +offset, host_ptr);
                });
            }

            void remapping_ptr_to_host() {
                for (auto elm : _address) {
                    ptr_t *point_to_ptr = reinterpret_cast<ptr_t *>(elm);

                    ptr_t ptr = *point_to_ptr;

                    if (is_host_ptr(ptr) || ptr == 0) {
                        continue;
                    }
                    auto host_ptr = to_host_ptr(ptr);
                    set_ptr_value(ptr_t(0), elm, host_ptr);
                }
            }

            void synchronize_all_to_host() {
                remapping_ptr_to_host();
                BaseClass::_device_buffer.download(_memory_block.template address<std::byte>(), _size_in_bytes);
            }

            void remapping_ptr_to_device(element_type &elm) {
                for_each_all_registered_member<T>([&](size_t offset, const char *name, auto ptr) {
                    ptr_t host_ptr = get_ptr_value(&elm, offset);
                    auto device_ptr = to_device_ptr(host_ptr);
                    set_ptr_value(&elm, offset, device_ptr);
                });
            }

            void remapping_ptr_to_device() {
                for (auto elm : _address) {
                    ptr_t *point_to_ptr = reinterpret_cast<ptr_t *>(elm);

                    ptr_t ptr = *point_to_ptr;

                    if (is_device_ptr(ptr) || ptr == 0) {
                        continue;
                    }
                    auto device_ptr = to_device_ptr(ptr);
                    set_ptr_value(ptr_t(0), elm, device_ptr);
                }
            }

            void synchronize_all_to_device() {
                remapping_ptr_to_device();
                BaseClass::_device_buffer.upload(_memory_block.template address<std::byte>(), _size_in_bytes);
            }
        };
    }
}