//
// Created by Zero on 2021/8/6.
//

#include "base_libs/lstd/lstd.h"
#include <algorithm>
#include <cstdlib>
#include "util/clock.h"
#include "iostream"
using namespace lstd;
using std::cout;
using std::endl;

class Base {
public:
    virtual int func() = 0;
};

class A : public Base {
public:
    int func() override {
        return rand();
    }
};

class B : public Base {
public:
    int func() override {
        return -rand();
    }
};

class E {
public:
    int func() {
        return rand();
    }
};

class C {
public:
    int func() {
        return -rand();
    }
};

class D {
public:
    int func() {
        return rand();
    }
};

class CD : public lstd::Variant<C, D> {
public:
    using Variant::Variant;

    int func() {
        LUMINOUS_VAR_DISPATCH(func)
    }
};

int main() {
    int a = 0;
    E ca;
//    std::cin >> a;
    Base *p;
    if (a % 2 == 0) {
        p = new A();
    } else {
        p = new B();
    }

    luminous::Clock clk;
    for (int i = 0; i < 10000000; ++i) {
//        a += p->func();
        a += ca.func();
    }
    cout << a << endl << clk.elapse_ms() << endl;

    CD cd;
    if (a % 2 == 0) {
        cd = CD(D());
    } else {
        cd = CD(C());
    }
    clk.start();
    a = 0;
    for (int i = 0; i < 10000000; ++i) {
        a += cd.func();
    }
    cout << a << endl << clk.elapse_ms() << endl;
}
