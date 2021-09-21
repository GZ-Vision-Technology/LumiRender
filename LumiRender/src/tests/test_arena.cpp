//
// Created by Zero on 12/09/2021.
//

#include <iostream>
#include "core/memory/allocator.h"
#include "render/lights/light.h"
#include "core/backend/managed.h"
using namespace std;
using namespace luminous;

class A {
public:
    int a;
    A(int a): a(a) {
        printf("adfaadfadsf\n");
    }






};

void test() {
    Managed<AreaLight, const AreaLight, Allocator<AreaLight>> a;
    Light l;
    AreaLight var(0,make_float3(),0,0);
    a.push_back(var);

    auto v = std::vector<A, Allocator<A>>();
    v.reserve(10);

    for (int i = 0; i < 9; ++i) {
        v.emplace_back(i);
    }

    for (int i = 0; i < 9; ++i) {
        cout << v[i].a << endl;
    }
}

void test2() {
    Managed<int> managed;
    int n = 1;
    managed.reserve(n);
    for (int i = 0; i < n; ++i) {
        managed.push_back(i);

    }
    cout << managed.size_in_bytes() << endl;
    cout << get_arena().usage() << endl;
    cout << sizeof (vector<int>) << endl;
}

int main() {

//    test();
    test2();


    return 0;
}