//
// Created by Zero on 2021/9/16.
//

#pragma once

#include <utility>
#include <iostream>

namespace luminous {
    inline namespace refl {

#define REFL_MAX_MEMBER_COUNT 128

        template<int N>
        struct Int : Int<N - 1> {
        };

        template<>
        struct Int<0> {
        };

        template<int N>
        struct Sizer {
            char _[N];
        };

#define REFL_CLASS(NAME)                                                        \
    using ReflSelf = NAME;                                                      \
    static refl::Sizer<1> _member_counter(...);                                 \
    template<int N>                                                             \
    struct MemberRegister {                                                     \
        template<typename F>                                                    \
        static void process(const F &f) {}                                      \
    };

#define DEFINE_AND_REGISTER_MEMBER(TYPE, NAME, ...)                             \
    TYPE NAME{__VA_ARGS__};                                                     \
    static constexpr int NAME##_refl_index =                                    \
        sizeof((_member_counter(                                                \
            (refl::Int<REFL_MAX_MEMBER_COUNT>*)nullptr)));                      \
    static_assert(NAME##_refl_index <= REFL_MAX_MEMBER_COUNT,                   \
                "index must not greater than REFL_MAX_MEMBER_COUNT");           \
    static refl::Sizer<NAME##_refl_index + 1>                                   \
        (_member_counter(refl::Int<NAME##_refl_index + 1> *));                  \
    template<>                                                                  \
    struct MemberRegister<NAME##_refl_index - 1> {                              \
        template<typename F>                                                    \
        static void process(const F &f) {                                       \
            f(&ReflSelf::NAME, #NAME);                                          \
        }                                                                       \
    };

        namespace detail {
            template<typename T, typename F, int...Is>
            void for_each_registered_member_aux(const F &f, std::integer_sequence<int, Is...>) {
                (T::template MemberRegister<Is>::template process<F>(f), ...);
            }

            template<typename T, typename F>
            void for_each_registered_member(const F &f) {
                for_each_registered_member_aux<T>(
                        f, std::make_integer_sequence<int, REFL_MAX_MEMBER_COUNT>());
            }

        }

        template<typename T, typename F>
        void for_each_registered_member(const F &f) {
#define OFFSET_OF(Class, member) reinterpret_cast<size_t>(&((*(Class *) 0).*member))
            detail::for_each_registered_member<T>([&](auto member_ptr, const char *name) {
                f(OFFSET_OF(T, member_ptr), name);
            });
#undef OFFSET_OF
        }

        template<typename...T>
        struct BaseBinder : public T ... {
            using Bases = std::tuple<T...>;
        };

        namespace detail {
            template<typename T, typename F, int...Is>
            void for_each_direct_base_aux(const F &f, std::integer_sequence<int, Is...>) {
                (f.template operator()<std::tuple_element_t<Is, typename T::Bases>>(), ...);
            }
        }

        template<typename T, typename F>
        void for_each_direct_base(const F &f) {
            detail::for_each_direct_base_aux<T>(
                    f, std::make_integer_sequence<int, std::tuple_size_v<typename T::Bases>>());
        }

        template<typename F>
        struct Visitor {
            F func;

            explicit Visitor(const F &f) : func(f) {}

            template<typename T>
            void operator()(T *ptr = nullptr) const {
                for_each_direct_base<T>(*this);
                func((T *) nullptr);
            }
        };

        template<typename T, typename F>
        void for_each_all_base(const F &f) {
            Visitor <F> visitor(f);
            for_each_direct_base<T>(visitor);
        }

        template<typename T, typename F>
        void for_each_all_registered_member(const F &func) {
            for_each_all_base<T>([&](auto ptr) {
                using Base = std::remove_pointer_t<decltype(ptr)>;
                for_each_registered_member<Base>([&](auto offset, auto member_name) {
                    func(offset, member_name, ptr);
                });
            });
            for_each_registered_member<T>([&](auto offset, auto member_name) {
                func(offset, member_name, (T *) nullptr);
            });
        }
    }
}