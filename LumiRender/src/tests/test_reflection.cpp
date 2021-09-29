#pragma once

#include "core/refl/factory.h"
#include "render/light_samplers/light_sampler.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

namespace luminous {

}

template<typename T>
struct A : public BaseBinder<> {

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
struct D : BASE_CLASS(B<T>, C, TT) {
    REFL_CLASS(D)

    DEFINE_AND_REGISTER_MEMBER(void *, pd);
};

class LS : public BaseBinder<Variant<UniformLightSampler>> {
public:
    using BaseBinder::BaseBinder;
    REFL_CLASS(LS)

    void print() {
        cout << "coao" << endl;
    }

    DEFINE_AND_REGISTER_MEMBER(void *, pd);
};

int main() {

//    for_each_registered_member<LightSampler>([&](auto offset, auto name) {
//        cout << "  " << name << "   " <<offset << endl;
//    });


//    for_each_all_registered_member<D<int>>([&](auto offset, auto name, auto ptr) {
//        cout << typeid(ptr).name() << "  " << name << "   " <<offset << endl;
//    });

//    for_each_all_base<D<int>>([&](auto p) {
//        using T = std::remove_pointer_t<decltype(p)>;
//        std::cout << typeid(T).name() << std::endl;
//    });

using T = D<int>;
//REGISTER(T)
//    for_each_all_registered_member<D<int>>([&](auto offset, auto name, auto ptr) {
//        cout << typeid(ptr).name() << "  " << name << endl;
//    });
}
