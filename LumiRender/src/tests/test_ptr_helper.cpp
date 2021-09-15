//
// Created by Zero on 12/09/2021.
//

#include <iostream>
#include "core/reflection.h"

using std::cout;
using std::endl;
using namespace luminous;

class A : public Object {
public:
    int b;
    A *pa{};
    using Super = Object;

    GEN_PTR_MEMBER_SIZE_FUNC(Object, pa)

    GEN_MAPPING_MEMBER_PTR_FUNC(pa)
};

class B : public A {
public:
    int b;
};

class C : public A {
public:
    int c;

    DEFINE_PTR_VARS(Object *pc, Object *p1)

    GEN_PTR_MEMBER_SIZE_FUNC(A, pc, p1)


};

REGISTER(B)
REGISTER(A)
REGISTER(C)

void test() {

}

int main() {

    test();


    return 0;
}