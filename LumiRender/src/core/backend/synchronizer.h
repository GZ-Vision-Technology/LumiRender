//
// Created by Zero on 29/09/2021.
//


#pragma once

#include "managed.h"
#include "ptr_mapper.h"
#include "core/memory/arena.h"

namespace luminous {
    inline namespace core {

        /**
         * synchronize between host and device
         * use for instance of class T, that T has pointer member
         * @tparam T
         */
        template<typename T>
        class Synchronizer {
        public:
            using element_type = T;
        private:
            MemoryBlock *_memory_block{};
            Device * _device{};
            Managed<T> _managed{_device};
        public:
            explicit Synchronizer(Device *device)
                : _device(device) {}

            void reserve(size_t n_element);

            void synchronize_to_device();

            void synchronize_to_host();

            void for_each_all_ptr_field() {

            }
        };
    }
}