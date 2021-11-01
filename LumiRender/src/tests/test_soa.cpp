//
// Created by Zero on 27/08/2021.
//

#include "render/integrators/wavefront/work_items.h"
#include "cpu/cpu_impl.h"
#include "gpu/framework/cuda_impl.h"
#include <vector>

using std::cout;
using std::endl;

using namespace luminous;

class Test {
public:
    float t{1};
    luminous::float3 pos{};


};


//LUMINOUS_SOA(Test, t, pos)


template<>
struct SOA<Test> {
public:
    static constexpr bool definitional = true;
    using element_type = Test;
    SOA() = default;
    int capacity;
    SOAMember<decltype(element_type::t), Device *>::type t;
    SOAMember<decltype(element_type::pos), Device *>::type pos;
    SOA(int n, Device *device) : capacity(n) {
        t = SOAMember<decltype(element_type::t), Device *>::create(n, device);
        pos = SOAMember<decltype(element_type::pos), Device *>::create(n, device);
    }
    SOA &operator=(const SOA &s) {
        capacity = s.capacity;
        this->t = s.t;
        this->pos = s.pos;
        return *this;
    }
    element_type operator[](int i) const {
        (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (24)), 0));;;
        element_type r;
        r.t = this->t[i];
        r.pos = this->pos[i];
        return r;
    }
    struct GetSetIndirector {
        SOA *soa;
        int i;
        operator element_type() const {
            element_type r;
            r.t = soa->t[i];
            r.pos = soa->pos[i];
            return r;
        }
        void operator=(const element_type &a) const {
            soa->t[i] = a.t;
            soa->pos[i] = a.pos;
        }
    };
    GetSetIndirector operator[](int i) {
        (void) ((!!(i < capacity)) || (_wassert(L"i < capacity", L"_file_name_", (unsigned) (24)), 0));;;
        return GetSetIndirector{this, i};
    }
    template<typename TDevice>
    SOA<element_type> to_host(TDevice *device) const {
        if (device->is_cpu()) { return *this; }
        auto ret = SOA<element_type>(capacity, device);
        ret.t = SOAMember<decltype(element_type::t), TDevice *>::clone_to_host(t, capacity, device);
        ret.pos = SOAMember<decltype(element_type::pos), TDevice *>::clone_to_host(pos, capacity, device);
        return ret;
    }
};

int main() {
    auto device = create_cpu_device();
//    auto cudevice = create_cuda_device();
    SOA<Test> st(3, device.get());

    Test t;

    auto p = make_float3(9.9);

    t.pos = p;

    cout << &t << endl;

    st[0] = t;

    Test ss = st[0];

    int i =0 ;

    return 0;
}