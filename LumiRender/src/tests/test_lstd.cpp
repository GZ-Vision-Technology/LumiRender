//
// Created by Zero on 2021/1/31.
//

#include "graphics/lstd/lstd.h"
#include "graphics/common.h"

using namespace luminous;

class Sub1 {
public:
    int fun1() {
        return 0;
    }

    int fun2(int a) {
        return a;
    }
};

class Sub2 {
public:
    int fun1() {
        return 1;
    }

    int fun2(int a) {
        return  2* a;
    }
};

class Base : public lstd::Variant<Sub1, Sub2> {
public:
    using Variant::Variant;
    int fun1() {
        return dispatch([](auto &&arg) { return arg.fun1(); });
    }

    int fun2(int a) {
        LUMINOUS_VAR_DISPATCH(fun2, a);
    }
};

class BaseP : public lstd::Variant<Sub1 *, Sub2 * > {
public:
    using Variant::Variant;
    int fun1() {
        return dispatch([](auto &&arg) { return arg->fun1(); });
    }

    int fun2(int a) {
        LUMINOUS_VAR_PTR_DISPATCH(fun2, a);
    }
};

using namespace std;

void testVariant() {
    Sub1 s1 = Sub1();
    Sub2 s2 = Sub2();

    cout << s1.fun1() << endl;
    cout << s2.fun1() << endl;

    Base b = s2;

    Base b2 = s1;

    cout << b.fun1() << endl;
    cout << b.fun2(9) << endl;

    BaseP bp = &s1;

    BaseP bp2 = &s2;

    cout << bp.fun1() << endl;
    cout << bp.fun2(9) << endl;

    cout << bp2.fun1() << endl;
    cout << bp2.fun2(9) << endl;
}

void test_math() {
//    Box3f box(make_float3(3.1));
//    cout << box.to_string() << endl;

//    auto a = int3x3(3.234f) ;
//    cout << a.to_string();

    auto m1 = make_double4x4(-2);

//    auto v1 = make_double4(1);
//    auto v2 = v1;
//    cout << m1.to_string();
//
//    auto v3 = v2 * v1;
//    cout << inverse(m1).to_string() << endl;
//    cout << transpose(m1).to_string() << endl;
    auto m2 = inverse(m1);

//    auto m3 =  m2 * m1;
    cout << m2.to_string() << endl;
}

void test_transform() {
    auto t = make_float3(1,2,3);
    auto tsf = Transform::translation(t);

    auto r = Transform::rotation(make_float3(3,1,2), 30);
    auto s = Transform::scale(make_float3(3,4,9));

    tsf = r * tsf * s;

    auto inv = tsf.inverse();

    auto p = make_float3(5,6,7);

    cout << tsf.mat4x4().to_string() << endl;
    auto np = tsf.apply_normal(p);
    auto nnp = inv.apply_normal(np);
    cout << p.to_string() << endl;
    cout << np.to_string() << endl;
    cout << nnp.to_string() << endl;

     np = tsf.apply_vector(p);
     nnp = inv.apply_vector(np);
    cout << p.to_string() << endl;
    cout << np.to_string() << endl;
    cout << nnp.to_string() << endl;

     np = tsf.apply_point(p);
     nnp = inv.apply_point(np);
    cout << p.to_string() << endl;
    cout << np.to_string() << endl;
    cout << nnp.to_string() << endl;

    cout << is_nan(0) << endl;
}

int main() {

//    testVariant();

//    test_math();
    test_transform();

    return 0;
}