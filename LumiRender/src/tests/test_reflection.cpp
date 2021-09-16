#pragma once

#include "core/refl/factory.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

class A : public Object {
public:
    REFL_CLASS(A)

    DEFINE_AND_REGISTER_MEMBER(Object*, a1)

    DEFINE_AND_REGISTER_MEMBER(Object *, a2)

    int iv = 0;
};

REGISTER(A)

class B : public A {
public:
    REFL_CLASS(B)

    int iv2 = 0;

    DEFINE_AND_REGISTER_MEMBER(Object *, a2)
};

REGISTER(B)

int main() {


    for_each_ptr_member<B>([&](auto offset, auto name) {
        cout << name << endl;
    });

}