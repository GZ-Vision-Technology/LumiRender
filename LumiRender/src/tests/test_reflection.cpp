#pragma once

#include <utility>
#include <iostream>

#define REFL_BEGIN namespace refl {
#define REFL_END   }

REFL_BEGIN

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
    static ::refl::Sizer<1> _member_counter(...);                               \
    template<int N>                                                             \
    struct MemberRegister                                                       \
    {                                                                           \
        template<typename F>                                                    \
        static void process(const F &f) {}                                      \
    };

#define DEFINE_MEMBER(TYPE, NAME)                                               \
    TYPE NAME;                                                                  \
    static constexpr int NAME##_refl_index =                                    \
        sizeof((_member_counter(                                                \
            (::refl::Int<REFL_MAX_MEMBER_COUNT>*)nullptr)));                    \
    static ::refl::Sizer<NAME##_refl_index + 1>                                 \
        (_member_counter(::refl::Int<NAME##_refl_index + 1> *));                \
    template<>                                                                  \
    struct MemberRegister<NAME##_refl_index - 1>                                \
    {                                                                           \
        template<typename F>                                                    \
        static void process(const F &f)                                         \
        {                                                                       \
            f(&ReflSelf::NAME, #NAME);                                          \
        }                                                                       \
    };

    template<typename T, typename F, int...Is>
    void forEachMemberAux(const F &f, std::integer_sequence<int, Is...>) {
        (T::template MemberRegister<Is>::template process<F>(f), ...);
    }

    template<typename T, typename F>
    void forEachMember(const F &f) {
        forEachMemberAux<T>(
                f, std::make_integer_sequence<int, REFL_MAX_MEMBER_COUNT>());
    }

REFL_END

struct A {
    REFL_CLASS(A)

    DEFINE_MEMBER(int, a)

    DEFINE_MEMBER(float, b)

    DEFINE_MEMBER(double, c)
};

int main() {
    refl::forEachMember<A>([&](auto member_ptr, const char *name) {
        std::cout << "name: " << name;
        std::cout << " , offset: " << reinterpret_cast<size_t>(&((*(A *) 0).*member_ptr));
        std::cout << " , size: " << sizeof((*(A *) 0).*member_ptr);
        std::cout << std::endl;
    });
}