//
// Created by Zero on 17/09/2021.
//


#pragma once

#include <vector>
#include "arena.h"

namespace luminous {
    inline namespace core {

        MemoryArena &get_arena();

        template<class Ty>
        class Allocator {
        public:
            MemoryArena &arena;
        public:
            static_assert(!std::is_const_v<Ty>, "The C++ Standard forbids containers of const elements "
                                                "because allocator<const T> is ill-formed.");

            using _From_primary = Allocator;

            using value_type = Ty;

            typedef Ty *pointer;
            typedef const Ty *const_pointer;

            typedef Ty &reference;
            typedef const Ty &const_reference;

            using size_type = size_t;
            using difference_type = ptrdiff_t;

            using propagate_on_container_move_assignment = std::true_type;
            using is_always_equal = std::true_type;

            template<class Other>
            struct rebind {
                using other = Allocator<Other>;
            };

            LM_NODISCARD Ty *address(Ty &Val) const noexcept {
                return _STD addressof(Val);
            }

            LM_NODISCARD const Ty *address(const Ty &Val) const noexcept {
                return _STD addressof(Val);
            }

            constexpr Allocator() noexcept: arena(get_arena()) {}

            constexpr Allocator(const Allocator &) noexcept = default;

            template<class Other>
            constexpr explicit Allocator(const Allocator<Other> &other) noexcept : arena(other.arena) {}

            void deallocate(Ty *const ptr, const size_t count) {

            }

            LM_NODISCARD __declspec(allocator) Ty *allocate(_CRT_GUARDOVERFLOW const size_t count) {
                return arena.template allocate<Ty>(count);
            }

            LM_NODISCARD __declspec(allocator) Ty *allocate(
                    _CRT_GUARDOVERFLOW const size_t count, const void *) {
                return allocate(count);
            }

            template<class Objty, class... Args>
            void construct(Objty *const ptr, Args &&... args) {
                construct_at(ptr, std::forward<Args>(args)...);
            }

            template<class Uty>
            void destroy(Uty *const _Ptr) {
                _Ptr->~Uty();
            }

            LM_NODISCARD size_t max_size() const noexcept {
                return static_cast<size_t>(-1) / sizeof(Ty);
            }
        };
    }
}