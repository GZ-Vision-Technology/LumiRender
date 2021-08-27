//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/macro_map.h"
#include "base_libs/header.h"
#include "core/backend/device.h"


namespace luminous {


    inline namespace render {
        template<typename T>
        struct SOA;

        template<>
        struct SOA<float3> {
        public:
            XPU SOA() = default;

            using element_type = float3;

            XPU SOA(size_t n, const std::shared_ptr<Device> &device) : size(n) {
                x = device->obtain_restrict_ptr<decltype(element_type::x)>(n);
                y = device->obtain_restrict_ptr<decltype(element_type::y)>(n);
                z = device->obtain_restrict_ptr<decltype(element_type::z)>(n);
            }

            XPU SOA &operator=(const SOA &s) {
                size = s.size;
                this->x = s.x;
                this->y = s.y;
                this->z = s.z;
                return *this;
            }

            XPU element_type operator[](int i) const {
                DCHECK_LT(i, size);
                element_type r;
                r.x = this->x[i];
                r.y = this->y[i];
                r.z = this->z[i];
                return r;
            }

            struct GetSetIndirector {
                XPU operator element_type() const {
                    element_type r;
                    r.x = soa->x[i];
                    r.y = soa->y[i];
                    r.z = soa->z[i];
                    return r;
                }

                XPU void operator=(const element_type &a) const {
                    soa->x[i] = a.x;
                    soa->y[i] = a.y;
                    soa->z[i] = a.z;
                }

                SOA *soa;
                int i;
            };

            XPU GetSetIndirector operator[](int i) {
                DCHECK_LT(i, size);
                return GetSetIndirector{this, i};
            }

            decltype(element_type::x) *LM_RESTRICT x;
            decltype(element_type::y) *LM_RESTRICT y;
            decltype(element_type::z) *LM_RESTRICT z;
            size_t size;
        };




#define LUMINOUS_SOA_BEGIN(StructName)  template<> \
struct SOA<StructName> {                           \
public:                                            \
using element_type = StructName;                   \
SOA() = default;                                   \
size_t size;

// member definition
#define LUMINOUS_SOA_MEMBER(MemberName) decltype(element_type::MemberName) *LM_RESTRICT MemberName;
#define LUMINOUS_SOA_MEMBERS(...) MAP(LUMINOUS_SOA_MEMBER,__VA_ARGS__)

// constructor definition
#define LUMINOUS_SOA_MEMBER_ASSIGNMENT(MemberName) MemberName = device->obtain_restrict_ptr<decltype(element_type::MemberName)>(n);
#define LUMINOUS_SOA_CONSTRUCTOR(...) \
SOA(size_t n, const std::shared_ptr<Device> &device) : size(n) { \
MAP(LUMINOUS_SOA_MEMBER_ASSIGNMENT,__VA_ARGS__) }


// assignment function definition
#define LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT(MemberName) this->MemberName = s.MemberName;
#define LUMINOUS_SOA_ASSIGNMENT(...)                            \
XPU SOA &operator=(const SOA &s) { size = s.size;               \
MAP(LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__) \
return *this; }


// access function definition
#define LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT(MemberName) r.MemberName = this->MemberName[i];
#define LUMINOUS_SOA_ACCESSOR(...) \
XPU element_type operator[](int i) const { DCHECK_LT(i, size); element_type r;        \
MAP(LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__) return r; }

// get set struct
#define LUMINOUS_SOA_GET_SET_CASTER_IMPL(MemberName) r.MemberName = soa->MemberName[i];
#define LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL(MemberName) soa->MemberName[i] = a.MemberName;
#define LUMINOUS_SOA_SET_GET_STRUCT(...) struct GetSetIndirector { SOA *soa;int i; \
XPU operator element_type() const { element_type r;                        \
MAP(LUMINOUS_SOA_GET_SET_CASTER_IMPL, __VA_ARGS__) return r;}                     \
XPU void operator=(const element_type &a) const {                                  \
MAP(LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL, __VA_ARGS__) }};

// get set accessor
#define LUMINOUS_SOA_INDIRECTOR_ACCESSOR XPU GetSetIndirector operator[](int i) { \
DCHECK_LT(i, size);                                                               \
return GetSetIndirector{this, i};}

#define LUMINOUS_SOA_END  };

#define LUMINOUS_SOA(StructName, ...) LUMINOUS_SOA_BEGIN(StructName)\
        LUMINOUS_SOA_MEMBERS(__VA_ARGS__)                           \
        LUMINOUS_SOA_CONSTRUCTOR(__VA_ARGS__)                       \
        LUMINOUS_SOA_ASSIGNMENT(__VA_ARGS__)                        \
        LUMINOUS_SOA_ACCESSOR(__VA_ARGS__)                          \
        LUMINOUS_SOA_SET_GET_STRUCT(__VA_ARGS__)                    \
        LUMINOUS_SOA_INDIRECTOR_ACCESSOR                            \
        LUMINOUS_SOA_END \


        LUMINOUS_SOA(float2, x, y)





    }
}