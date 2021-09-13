//
// Created by Zero on 12/09/2021.
//

#include <iostream>
#include "core/reflection.h"

using std::cout;
using std::endl;

using namespace luminous;

//using

class A : public Object {
public:
    int b;
    A *pa{};
    using Super = Object;

    GEN_PTR_MEMBER_SIZE_FUNC(pa)
};


class B : public A {
public:
    int b;
};

class C : public A {
public:
    int c;
    A *pc{};
    using Super = A;

//    [[nodiscard]]size_t ptr_member_size() const override {
//        return (+((pc) ? (pc)->
//                real_size() :
//                  0)) + A::ptr_member_size();
//    }
//    B *p2;
    GEN_PTR_MEMBER_SIZE_FUNC(pc)
};

REGISTER(B)
REGISTER(A)
REGISTER(C)

void test_reflection() {
    Object object;
    A a;
    C *p = new C;
//    p = new B;
    p->pc = new A;
    p->pa = new A;

    auto cf = ClassFactory::instance();
    cout << "object size:" << sizeof(Object) << endl;
    cout << "C size:" << cf->size_of(new C) << endl;
    cout << "A size:" << cf->size_of(new A) << endl;
    cout << "B size:" << cf->size_of(new B) << endl;

    cout << "C object:" << p->real_size();
}

int main() {

//    test_crtp();

    test_reflection();

    return 0;
}