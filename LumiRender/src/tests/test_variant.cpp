//
// Created by Zero on 2021/8/6.
//

#include "base_libs/lstd/lstd.h"
#include <algorithm>
#include <cstdlib>
#include "util/clock.h"
#include "iostream"
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

class CD : public luminous::lstd::Variant<C, D> {
public:
    using Variant::Variant;

    int func() {
        LUMINOUS_VAR_DISPATCH(func)
    }
};

void test_normal(int a) {
    E ca;
    Base *p;
    a = 0;
    luminous::Clock clk;
    for (int i = 0; i < 10000000; ++i) {
        a += ca.func();
    }
    cout << a << endl << clk.elapse_ms() << endl;
}

void test_virtual(int a) {
    E ca;

    Base *p;
    if (a % 2 == 0) {
        p = new A();
    } else {
        p = new B();
    }
    a = 0;
    luminous::Clock clk;
    for (int i = 0; i < 10000000; ++i) {
        a += p->func();
    }
    cout << a << endl << clk.elapse_ms() << endl;
}

void test_variant(int a) {
    CD cd;
    if (a % 2 == 0) {
        cd = CD(D());
    } else {
        cd = CD(C());
    }
    luminous::Clock clk;
    clk.start();
    a = 0;
    for (int i = 0; i < 10000000; ++i) {
        a += cd.func();
    }
    cout << a << endl << clk.elapse_ms() << endl;
}

int main() {
    int a = 0;
    std::cin >> a;

    test_virtual(a);
    test_variant(a);
    test_normal(a);
}
