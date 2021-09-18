//
// Created by Zero on 12/09/2021.
//

#include <iostream>
#include "core/memory/allocator.h"
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
    std::vector<int> a;
    a.push_back(1);


    auto v = std::vector<A, Allocator<A>>();
    v.reserve(10);

    for (int i = 0; i < 9; ++i) {
        v.emplace_back(i);
    }

    for (int i = 0; i < 9; ++i) {
        cout << v[i].a << endl;
    }
}

int main() {

    test();


    return 0;
}