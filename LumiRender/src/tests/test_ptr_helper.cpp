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
};

class T {
    virtual int func() = 0;
};

class B : public A {
public:
    int b;
};

class C : public Object {
public:
    int c;
    A *p{};
//    B *p2;
    GEN_REAL_SIZE_FUNC(p)
};

REGISTER(B)
REGISTER(A)
REGISTER(C)

void test_reflection() {
    Object object;
    A a;
    C *p = new C;
//    p = new B;
p->p = new A;
//p->p2 = new B;

    auto cf = ClassFactory::instance();
    cout <<"object size:" << sizeof(Object) << endl;
    cout <<"C size:" << cf->size_of(new C) << endl;
//    cout <<"A size:" << cf->size_of(new A) << endl;
//    cout <<"B size:" << cf->size_of(new B) << endl;
    cout << "p size:" << p->real_size() << endl;
    cout << "p size:" << (reinterpret_cast<Object*>(p))->real_size() << endl;
}

int main() {

//    test_crtp();

    test_reflection();

    return 0;
}