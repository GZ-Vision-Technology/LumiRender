#pragma once

#include <type_traits>
#include <iostream>

using namespace std;

#define REFL_BEGIN namespace refl {
#define REFL_END   }

REFL_BEGIN

    namespace detail {
        struct ForEachMemberVariable;
    }

#define BEGIN_CLASS(NAME) BEGIN_CLASS_IMPL(NAME, __LINE__)
#define BEGIN_CLASS_IMPL(NAME, LINE)                                            \
    class NAME                                                                  \
    {                                                                           \
        using Self = NAME;                                                      \
        static constexpr int BEGIN_LINE = LINE;                                 \
        friend struct ::refl::detail::ForEachMemberVariable;                    \
        template<int L>                                                         \
        struct RegisterMemberVariable                                           \
        {                                                                       \
            template<typename F>                                                \
            static void process(const F &f)                                     \
            {                                                                   \
            }                                                                   \
        };

#define DEFINE_MEMBER(TYPE, NAME) DEFINE_MEMBER_IMPL(TYPE, NAME, __LINE__)
#define DEFINE_MEMBER_IMPL(TYPE, NAME, LINE)                                    \
    TYPE NAME;                                                                  \
    template<>                                                                  \
    struct RegisterMemberVariable<LINE>                                         \
    {                                                                           \
        template<typename F>                                                    \
        static void process(const F &f)                                         \
        {                                                                       \
            f(&Self::NAME, #NAME);                                              \
        }                                                                       \
    };

#define END_CLASS(NAME) END_CLASS_IMPL(NAME, __LINE__)
#define END_CLASS_IMPL(NAME, LINE)                                              \
        static constexpr int END_LINE = LINE;                                   \
    };

    namespace detail {
        struct ForEachMemberVariable {
            template<typename T, typename F, int IBeg, int...Is>
            static void process_impl(const F &f, std::integer_sequence<int, Is...>) {
                (T::template RegisterMemberVariable<IBeg + Is>::template process(f), ...);
            }

            template<typename T, typename F>
            static void process(const F &f) {
                constexpr int IBeg = T::BEGIN_LINE;
                constexpr int IEnd = T::END_LINE;
                ForEachMemberVariable::process_impl<T, F, IBeg>(
                        f, std::make_integer_sequence<int, IEnd - IBeg>());
            }
        };
    }

    template<typename T, typename F>
    void forEachMemberVariable(const F &f) {
        detail::ForEachMemberVariable::process<T>(f);
    }

REFL_END

BEGIN_CLASS(A)

    DEFINE_MEMBER(int, a)

    DEFINE_MEMBER(float, b)

END_CLASS(A)

int main() {
    refl::forEachMemberVariable<A>(
            [&](auto member_ptr, const char *name) {
                std::cout << name << std::endl;
            });
    return 0;
}

