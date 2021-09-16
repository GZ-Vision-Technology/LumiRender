#pragma once

#include "src/core/refl/factory.h"

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

    void fun() {
        int _count = 0;
        refl::for_each_registered_member(*this, [&](auto obj, Object *ptr, auto name) {
            ++ _count;
        });
        _count = 0;
        _objects.reset(new Object_ptr[_count]);
        refl::for_each_registered_member(*this, [&](auto obj, Object *ptr, auto name) {
            _objects.get()[_count++] = ptr;
        });
    }
};

REGISTER(A)

class B : public A {
public:
    REFL_CLASS(B)

    int iv2 = 0;


};

REGISTER(B)

int main() {

    A a;

    a.iv = 9;
    A* pa = new A();
    a.a1 = pa;
    a.a2 = pa;


    a.fun();
//
////    a.a1 = pa;
////    a.a2 = new B;
////    cout << pa << endl;
////
//    cout << sizeof(std::unique_ptr<Object*>) << endl;
//    cout << sizeof(std::vector<Object*>) << endl;
//
//    refl::for_each_registered_member(a, [&](auto obj, Object *ptr, auto name) {
//        cout << "name : " << name << ",ptr:" << ptr << "  " << ptr->self_size() << endl;
//    });
}