#pragma once

#include "core/refl/factory.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

namespace luminous {
    inline namespace refl {
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
            Visitor<F> visitor(f);
            for_each_direct_base<T>(visitor);
        }

        template<typename T, typename F>
        void for_each_all_registered_member(const F &func) {
            for_each_all_base<T>([&](auto ptr) {
                using Base = std::remove_pointer_t<decltype(ptr)>;
                for_each_registered_member<Base>(func);
            });
            for_each_registered_member<T>(func);
        }
    }
}

template<typename T>
struct A : public BaseBinder<>{

    REFL_CLASS(A)

    DEFINE_AND_REGISTER_MEMBER(void *, pa);
};

template<typename T>
struct B : BaseBinder<A<T>> {
    REFL_CLASS(B)

    DEFINE_AND_REGISTER_MEMBER(void *, pb);
};

struct C : BaseBinder<> {
    REFL_CLASS(C)
    DEFINE_AND_REGISTER_MEMBER(void *, pc);
};

struct TT : BaseBinder<> {
    REFL_CLASS(TT)

    DEFINE_AND_REGISTER_MEMBER(void *, pt);
};

template<typename T>
struct D : BaseBinder<B<T>, C, TT> {
    REFL_CLASS(D)
    DEFINE_AND_REGISTER_MEMBER(void *, pd);
};

int main() {

//    for_each_all_base<D<int>>([&](auto p) {
//        using T = std::remove_pointer_t<decltype(p)>;
//        std::cout << typeid(T).name() << std::endl;
//    });

    for_each_all_registered_member<D<int>>([&](auto offset, auto name) {
        cout << name << endl;
    });
}
