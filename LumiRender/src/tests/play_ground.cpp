#include "util/parallel.h"

#include "iostream"

using namespace luminous;

using namespace std;

struct A {

};

void func(const A *p1, A *p2) {
    printf("p1 = %p, p2 = %p\n", p1, p2);
}

template<typename ...Args>
void func2(Args&&...args) {
    auto l = [&](uint ,uint) {
        func(std::forward<Args>(args)...);
    };

    l(0,0);
    luminous::parallel_for(1000, [&](uint ,uint) {
        func(std::forward<Args>(args)...);
    });
}

int main() {

    auto pa = new A;

    func2(pa, pa);

}