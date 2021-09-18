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

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef Ty *pointer;
            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef const Ty *const_pointer;

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef Ty &reference;
            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef const Ty &const_reference;

            using size_type = size_t;
            using difference_type = ptrdiff_t;

            using propagate_on_container_move_assignment = std::true_type;
            using is_always_equal = std::true_type;

            template<class Other>
            struct _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS rebind {
                using other = Allocator<Other>;
            };

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD Ty *address(Ty &Val) const noexcept {
                return _STD addressof(Val);
            }

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD const Ty *address(const Ty &Val) const noexcept {
                return _STD addressof(Val);
            }

            constexpr Allocator() noexcept: arena(get_arena()) {}

            constexpr Allocator(const Allocator &) noexcept = default;

            template<class _Other>
            constexpr Allocator(const Allocator<_Other> &other) noexcept : arena(other.arena) {}

            void deallocate(Ty *const ptr, const size_t count) {

            }

            _NODISCARD __declspec(allocator) Ty *allocate(_CRT_GUARDOVERFLOW const size_t count) {
                return arena.template allocate<Ty>(count);
            }

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD __declspec(allocator) Ty *allocate(
                    _CRT_GUARDOVERFLOW const size_t count, const void *) {
                return allocate(count);
            }

            template<class Objty, class... Args>
            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS void construct(Objty *const ptr, Args &&... args) {
                construct_at(ptr, std::forward<Args>(args)...);
            }

            template<class _Uty>
            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS void destroy(_Uty *const _Ptr) {
                _Ptr->~_Uty();
            }

            _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD size_t max_size() const noexcept {
                return static_cast<size_t>(-1) / sizeof(Ty);
            }
        };

//        template<typename T>
//        class Allocator : public std::allocator<T> {
//        private:
//            MemoryArena &_arena;
//        public:
//            using value_type = T;
//            using pointer = T *;
//            using const_pointer = const T *;
//            using void_pointer = void *;
//            using const_void_pointer = const void *;
//            using size_type = size_t;
//            using difference_type = std::ptrdiff_t;
//
//        public:
//            Allocator() : _arena(arena()) {}
//
//            explicit Allocator(MemoryArena &arena) : _arena(arena) {}
//
//            Allocator(Allocator &&other) noexcept: _arena(other._arena) {}
//
////            template<typename U>
////            struct rebind {
////                using other = Allocator<U>;
////            };
//
//            NDSC pointer allocate(size_t n) {
//                printf("Adsfdasf");
//                return _arena.template allocate<T>(n);
//            }
//
//            void deallocate(T *p, size_t n) {}
//
//            template<typename U, typename ...Args>
//            void construct(U *p, Args ... args) {
//                construct_at(p, std::forward<Args>(args)...);
//            }
//
//            template<typename Uty>
//            void destroy(Uty *p) {
//                p->~_Uty();
//            }
//
//            ~Allocator() = default;
//        };
    }
}