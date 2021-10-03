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
        class Synchronizer : public Managed<T> {
        public:
            using element_type = T;
        private:
            MemoryBlock *_memory_block{};
        public:
            explicit Synchronizer(Device *device)
                    : Managed<T>(device) {}

            void init(size_t n_element = 1) {
                static constexpr auto size = lstd::Sizer<T>::max_size;
                _memory_block = get_arena().create_memory_block_and_focus(size * n_element);
                Managed<T>::reserve(n_element);
                allocate_device(size);
//                Managed<T>::_device_buffer = Managed<T>::_device->create_buffer(size);
            }

            template<typename U>
            void add_element(const U &config) {
                auto elm = render::detail::create_ptr<T>(config);
                Managed<T>::push_back(elm);
            }

            void remapping_ptr_field() {

            }

            void for_each_all_ptr_field() {
                for (int i = 0; i < Managed<T>::size(); ++i) {
                    for_each_all_registered_member<T>([&](size_t offset, char *name, auto ptr) {
                        auto &elm = Managed<T>::at(i);
//                        set_ptr_value(&elm, )
                    });
                }
            }

            void synchronize_all_to_host() {

                Managed<T>::synchronize_to_host();
            }

            void synchronize_all_to_device() {

                Managed<T>::synchronize_to_device();
            }
        };
    }
}