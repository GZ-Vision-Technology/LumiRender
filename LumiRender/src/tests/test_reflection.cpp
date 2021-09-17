#pragma once

#include "core/refl/factory.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

class A : public Object {
public:
    REFL_CLASS(A)

    DECLARE_SUPER(Object);

    DEFINE_AND_REGISTER_MEMBER(Object*, a1)

//    DEFINE_AND_REGISTER_MEMBER(Object *, a2)

    int iv = 0;
};

//REGISTER(A)

RegisterAction<A> RegisterA;

int main() {

    auto a = new A;
    auto a1 = new A;

//    a->a1 = a;

    cout << ClassFactory::instance()->size_of(type_name(a)) << endl;

}