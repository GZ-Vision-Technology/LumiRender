//
// Created by Zero on 27/08/2021.
//

#include "render/integrators/wavefront/work_items.h"
#include "cpu/cpu_impl.h"
#include <vector>

using std::cout;
using std::endl;

using namespace luminous;

class Test {
public:
    float t{1};
    luminous::float3 pos{};


};


LUMINOUS_SOA(Test, t, pos)

template<typename... Args>
void func(Args&&... args) {


}

int main() {
    auto device = create_cpu_device();

    SOA<Test> st(3, device.get());

    Test t;

    cout << &t << endl;
    func(t, "Adsfasdf");

//    st[0] = t;
//
//    Test ss = st[1];

    return 0;
}