//
// Created by Zero on 27/08/2021.
//


#pragma once

#include "core/macro_map.h"

namespace luminous {
    inline namespace render {
        template<typename T>
        struct SOA {
            static constexpr bool definitional = false;
        };

        template<typename T, typename TDevice>
        struct SOAMember {
            static auto create(int n, const TDevice &device) {
                if constexpr (SOA<T>::definitional) {
                    return SOA<T>(n, device);
                } else {
                    return device->template obtain_restrict_ptr<T>(n);
                }
            }

            using type = decltype(create(0, nullptr));
        };
    }
}

#define LUMINOUS_SOA_BEGIN(StructName)  template<> \
struct SOA<StructName> {                           \
public:                                            \
static constexpr bool definitional = true;         \
using element_type = StructName;                   \
SOA() = default;                                   \
int capacity;

// todo add LM_RESTRICT
// member definition
#define LUMINOUS_SOA_MEMBER(MemberName) SOAMember<decltype(element_type::MemberName),std::shared_ptr<Device>>::type (MemberName);
#define LUMINOUS_SOA_MEMBERS(...) MAP(LUMINOUS_SOA_MEMBER,__VA_ARGS__)

// constructor definition
#define LUMINOUS_SOA_MEMBER_ASSIGNMENT(MemberName) MemberName = SOAMember<decltype(element_type::MemberName),std::shared_ptr<Device>>::create(n, device);
#define LUMINOUS_SOA_CONSTRUCTOR(...)                             \
SOA(int n, const std::shared_ptr<Device> &device) : capacity(n) { \
MAP(LUMINOUS_SOA_MEMBER_ASSIGNMENT,__VA_ARGS__) }


// assignment function definition
#define LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT(MemberName) this->MemberName = s.MemberName;
#define LUMINOUS_SOA_ASSIGNMENT(...)                            \
XPU SOA &operator=(const SOA &s) { capacity = s.capacity;       \
MAP(LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__) \
return *this; }


// access function definition
#define LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT(MemberName) r.MemberName = this->MemberName[i];
#define LUMINOUS_SOA_ACCESSOR(...)                                                  \
XPU element_type operator[](int i) const { DCHECK_LT(i, capacity); element_type r;  \
MAP(LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__) return r; }

// get set struct
#define LUMINOUS_SOA_GET_SET_CASTER_IMPL(MemberName) r.MemberName = soa->MemberName[i];
#define LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL(MemberName) soa->MemberName[i] = a.MemberName;
#define LUMINOUS_SOA_SET_GET_STRUCT(...) struct GetSetIndirector { SOA *soa;int i; \
XPU operator element_type() const { element_type r;                                \
MAP(LUMINOUS_SOA_GET_SET_CASTER_IMPL, __VA_ARGS__) return r;}                      \
XPU void operator=(const element_type &a) const {                                  \
MAP(LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL, __VA_ARGS__) }};

// get set accessor
#define LUMINOUS_SOA_INDIRECTOR_ACCESSOR XPU GetSetIndirector operator[](int i) { \
DCHECK_LT(i, capacity);                                                           \
return GetSetIndirector{this, i};}

#define LUMINOUS_SOA_END  };

#define MAKE_SOA_FRIEND(ClassName) friend class SOA<ClassName>;

#define LUMINOUS_SOA(StructName, ...) LUMINOUS_SOA_BEGIN(StructName)\
        LUMINOUS_SOA_MEMBERS(__VA_ARGS__)                           \
        LUMINOUS_SOA_CONSTRUCTOR(__VA_ARGS__)                       \
        LUMINOUS_SOA_ASSIGNMENT(__VA_ARGS__)                        \
        LUMINOUS_SOA_ACCESSOR(__VA_ARGS__)                          \
        LUMINOUS_SOA_SET_GET_STRUCT(__VA_ARGS__)                    \
        LUMINOUS_SOA_INDIRECTOR_ACCESSOR                            \
        LUMINOUS_SOA_END


