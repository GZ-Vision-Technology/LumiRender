//
// Created by Zero on 29/09/2021.
//


#pragma once

#include "managed.h"
#include "ptr_mapper.h"
#include "core/memory/arena.h"
#include "core/refl/reflection.h"
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
            size_t _size_in_bytes{};
        public:
            explicit Synchronizer(Device *device)
                    : BaseClass(device) {}

            void init(size_t n_element = 1, size_t size = lstd::Sizer<T>::max_size) {
                _size_in_bytes = size * n_element;
                USE_BLOCK(&_memory_block);
                _memory_block.allocate(_size_in_bytes);
                BaseClass::reserve(n_element);
                BaseClass::allocate_device(_size_in_bytes);
                PtrMapper::instance()->add_pair(_memory_block.interval_allocated(), BaseClass::device_interval());
            }

            template<typename U>
            void add_element(const U &config) {
                USE_BLOCK(&_memory_block);
                auto elm = render::detail::create_ptr<T>(config);
                BaseClass::push_back(elm);
            }

            LM_NODISCARD const element_type *device_ptr() const {
                return BaseClass::_device_buffer.template ptr<const element_type *>();
            }

            LM_NODISCARD element_type *device_ptr() {
                return BaseClass::_device_buffer.template ptr<element_type *>();
            }

            LM_NODISCARD BufferView<element_type> device_buffer_view(size_t offset = 0, size_t count = -1) {
                count = fix_count(offset, count, BaseClass::size());
                return BufferView<element_type>(BaseClass::_device_buffer.template ptr<element_type *>() + offset, count);
            }

            LM_NODISCARD BufferView<const element_type> const_device_buffer_view(size_t offset = 0, size_t count = -1) const {
                count = fix_count(offset, count, BaseClass::size());
                return BufferView<const element_type>(BaseClass::_device_buffer.template ptr<element_type *>() + offset, count);
            }

            void remapping_ptr_to_device() {
                for (int i = 0; i < BaseClass::size(); ++i) {
                    for_each_all_registered_member<T>([&](size_t offset, const char *name, auto ptr) {
                        auto &elm = BaseClass::at(i);
                        ptr_t host_ptr = get_ptr_value(&elm, offset);
                        auto device_ptr = PtrMapper::instance()->get_device_ptr(host_ptr);
                        set_ptr_value(&elm, offset, device_ptr);
                    });
                }
            }

            void synchronize_all_to_host() {

            }

            void synchronize_all_to_device() {
                remapping_ptr_to_device();
                BaseClass::_device_buffer.upload(_memory_block.template address<std::byte>(), _size_in_bytes);
            }
        };
    }
}