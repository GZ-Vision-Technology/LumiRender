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

class B : public A {
public:
    int b;
};

REGISTER(B)
REGISTER(A)

void test_reflection() {
    Object object;
    A a;
    Object *p = new A;
    cout << p->class_name() << endl;
    p = new B;
    cout << p->class_name() << endl;

    auto cf = ClassFactory::instance();

    cout << cf->size_of(new A) << endl;
    cout << cf->size_of(new B) << endl;
}

int main() {

//    test_crtp();

    test_reflection();

    return 0;
}