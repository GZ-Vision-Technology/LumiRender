#pragma once

#include "core/refl/factory.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

class A : public Object {
public:
    REFL_CLASS(A)

    int a0;

    DEFINE_AND_REGISTER_MEMBER(Object*, a1)

//    DEFINE_AND_REGISTER_MEMBER(Object *, a2)

    int iv = 0;
};

//REGISTER(A)

RegisterAction<A> RegisterA;

int main() {

    auto a = new A;
    auto a1 = new A;

    a->a1 = a1;

    refl::for_each_ptr_member<A>([&](uint64_t offset, auto name) {
        cout << offset << "  " << name << endl;
    });

    cout << uint64_t (a1) << endl;
    cout << a->get_value(8) << endl;

    a->set_value(8, 15);

    cout << a->a1 << endl;

    cout << a->get_value(8) << endl;

}