#pragma once

#include "core/refl/factory.h"
#include "render/light_samplers/light_sampler.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace luminous;

namespace luminous {

}

template<typename T>
struct A : public BaseBinder<> {

    REFL_CLASS(A)

    DEFINE_AND_REGISTER_MEMBER(void *, pa);
};

template<typename T>
struct B : BaseBinder<A<T>> {
    REFL_CLASS(B)

    DEFINE_AND_REGISTER_MEMBER(void *, pb);
};

struct C : BaseBinder<> {
    REFL_CLASS(C)

    DEFINE_AND_REGISTER_MEMBER(void *, pc);
};

struct TT : BaseBinder<> {
    REFL_CLASS(TT)

    DEFINE_AND_REGISTER_MEMBER(void *, pt);
};

template<typename T>
struct D : BASE_CLASS(B<T>, C, TT) {
    REFL_CLASS(D)

    DEFINE_AND_REGISTER_MEMBER(void *, pd);
};

#if 0
class LS : public BaseBinder<Variant<UniformLightSampler>> {
public:
    using BaseBinder::BaseBinder;
    REFL_CLASS(LS)

    void print() {
        cout << "coao" << endl;
    }

    DEFINE_AND_REGISTER_MEMBER(void *, pd);
};
#endif

namespace test_reflection {

class OffsetCalc;

class A {};

class A2 {

    friend class OffsetCalc;

    DECLARE_MEMBER_MAP(A2)

private:
    int *a;
    int *b;
};

BEGIN_MEMBER_MAP(A2)
MEMBER_MAP_ENTRY(a)
MEMBER_MAP_ENTRY(b)
END_MEMBER_MAP()

class B : A {

    friend class OffsetCalc;

    DECLARE_REFLECTION(B, A)

    DECLARE_MEMBER_MAP(B)

private:
    int *a_in_B;
    float *b_in_B;
};

BEGIN_MEMBER_MAP(B)
MEMBER_MAP_ENTRY(a_in_B)
MEMBER_MAP_ENTRY(b_in_B)
END_MEMBER_MAP()

class C : A2 {

    friend class OffsetCalc;

    DECLARE_REFLECTION(C, A2)

    DECLARE_MEMBER_MAP(C)

private:
    int *a_in_C;
    float *b_in_C;
};

BEGIN_MEMBER_MAP(C)
MEMBER_MAP_ENTRY(a_in_C)
MEMBER_MAP_ENTRY(b_in_C)
END_MEMBER_MAP()

class D : B, C {

    friend class OffsetCalc;

    DECLARE_REFLECTION(D, B, C)

    DECLARE_MEMBER_MAP(D)

private:
    int *a_in_D;
    float *b_in_D;
};

BEGIN_MEMBER_MAP(D)
MEMBER_MAP_ENTRY(a_in_D)
MEMBER_MAP_ENTRY(b_in_D)
END_MEMBER_MAP()

class OffsetCalc {
public:
  template<class Type>
  static void check_offset(const char *name, size_t base_offset, size_t field_offset) {
#define TEST_DCHECK_EQ_RET(a, b)                          \
    assert((a) == (b) && #a "is not coincident with" #b); \
    return


      if (strcmp(name, "a") == 0) {
          TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, a));
      } else if (strcmp(name, "b") == 0) {
          TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, b));
      } else if (strcmp(name, "a_in_B") == 0) {
          if constexpr (std::is_same_v<Type, B> || std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, a_in_B));
          }
      } else if (strcmp(name, "b_in_B") == 0) {
          if constexpr (std::is_same_v<Type, B> || std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, b_in_B));
          }
      } else if (strcmp(name, "a_in_C") == 0) {
          if constexpr (std::is_same_v<Type, C> || std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, a_in_C));
          }
      } else if (strcmp(name, "b_in_C") == 0) {
          if constexpr (std::is_same_v<Type, C> || std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, b_in_C));
          }
      } else if (strcmp(name, "a_in_D") == 0) {
          if constexpr (std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, a_in_D));
          }
      } else if (strcmp(name, "b_in_D") == 0) {
          if constexpr (std::is_same_v<Type, D>) {
              TEST_DCHECK_EQ_RET(field_offset - base_offset, offsetof(Type, b_in_D));
          }
      }

      assert("Unknown type");
#undef TEST_DCHECK_EQ_RET
  }
};


int test_reflection() {

    std::vector<std::pair<std::string, const void **>> offsets;
    luminous::refl::runtime_class<D>::visit_member_map(
            0, [&offsets](const char *name, size_t offset) {
                offsets.emplace_back(name, (const void **) offset);
            });

    std::cout << "_Runtime_class<D>" << std::endl;
    for (auto &field_entry : offsets) {
        std::cout << "name: " << field_entry.first
                  << ", offset address: " << field_entry.second << std::endl;
        OffsetCalc::check_offset<D>(field_entry.first.c_str(), 0, (size_t)field_entry.second);
    }

    offsets.clear();

    luminous::refl::runtime_class<C>::visit_member_map(
            0, [&offsets](const char *name, size_t offset) {
                offsets.emplace_back(name, (const void **) offset);
            });
    std::cout << "_Runtime_class<C>" << std::endl;
    for (auto &field_entry : offsets) {
        std::cout << "name: " << field_entry.first
                  << ", offset address: " << field_entry.second << std::endl;
        OffsetCalc::check_offset<C>(field_entry.first.c_str(), 0, (size_t)field_entry.second);
    }

    offsets.clear();

    luminous::refl::runtime_class<A2>::visit_member_map(
            0, [&offsets](const char *name, size_t offset) {
                offsets.emplace_back(name, (const void **) offset);
            });
    std::cout << "_Runtime_class<A2>" << std::endl;
    for (auto &field_entry : offsets) {
        std::cout << "name: " << field_entry.first
                  << ", offset address: " << field_entry.second << std::endl;
        OffsetCalc::check_offset<A2>(field_entry.first.c_str(), 0, (size_t)field_entry.second);
    }

    return 0;
}
};


int main() {

//    for_each_registered_member<LightSampler>([&](auto offset, auto name) {
//        cout << "  " << name << "   " <<offset << endl;
//    });


//    for_each_all_registered_member<D<int>>([&](auto offset, auto name, auto ptr) {
//        cout << typeid(ptr).name() << "  " << name << "   " <<offset << endl;
//    });

//    for_each_all_base<D<int>>([&](auto p) {
//        using T = std::remove_pointer_t<decltype(p)>;
//        std::cout << typeid(T).name() << std::endl;
//    });

using T = D<int>;
//REGISTER(T)
//    for_each_all_registered_member<D<int>>([&](auto offset, auto name, auto ptr) {
//        cout << typeid(ptr).name() << "  " << name << endl;
//    });

test_reflection::test_reflection();
}
