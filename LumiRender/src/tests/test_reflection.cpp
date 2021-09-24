#pragma once

#include "core/refl/factory.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

template<typename...Bases>
struct RegisterBase : Bases ... {
    using Ts = std::tuple<Bases...>;
};

template<typename T>
struct A : public RegisterBase<>{

    REFL_CLASS(A)

    DEFINE_AND_REGISTER_MEMBER(void *, pa);
};

template<typename T>
struct B : RegisterBase<A<T>> {
    REFL_CLASS(B)

    DEFINE_AND_REGISTER_MEMBER(void *, pb);
};

struct C : RegisterBase<> {
    REFL_CLASS(C)
    DEFINE_AND_REGISTER_MEMBER(void *, pc);
};

template<typename T>
struct D : RegisterBase<B<T>, C> {

};

template<typename T, typename F, int...Is>
void forEachDirectBaseAux(const F &f, std::integer_sequence<int, Is...>) {
    (f.template operator()<std::tuple_element_t<Is, typename T::Ts>>(), ...);
}

template<typename T, typename F>
void forEachDirectBase(const F &f) {
    forEachDirectBaseAux<T>(
            f, std::make_integer_sequence<int, std::tuple_size_v<typename T::Ts>>());
}

struct Visitor {
    template<typename T>
    void operator()() const {
        forEachDirectBase<T>(*this);
        std::cout << typeid(T).name() << std::endl;
        for_each_registered_member<T>([&](auto offset, auto name) {
            cout << name << endl;
        });
    }
};

struct TT : RegisterBase<C> {

};

int main() {

    Visitor visitor;

    forEachDirectBase<D<int>>(visitor);
}
