//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/macro_map.h"
#include "base_libs/header.h"
#include "core/backend/device.h"

#define PRINT(a) printf(#a": %d", a);

namespace luminous {


    inline namespace render {
        template<typename T>
        struct SOA;

        template<>
        struct SOA<float3> {
        public:
            XPU SOA() = default;

            XPU SOA(size_t n, const std::shared_ptr<Device> &device) : size(n) {
                x = device->obtain_restrict_ptr<decltype(float3::x)>(n);
                y = device->obtain_restrict_ptr<decltype(float3::y)>(n);
                z = device->obtain_restrict_ptr<decltype(float3::z)>(n);
            }

            XPU SOA &operator=(const SOA &s) {
                size = s.size;
                this->x = s.x;
                this->y = s.y;
                this->z = s.z;
                return *this;
            }

            XPU float3 operator[](int i) const {
                DCHECK_LT(i, size);
                float3 r;
                r.x = this->x[i];
                r.y = this->y[i];
                r.z = this->z[i];
                return r;
            }

            struct GetSetIndirector {
                XPU operator float3() const {
                    float3 r;
                    r.x = soa->x[i];
                    r.y = soa->y[i];
                    r.z = soa->z[i];
                    return r;
                }
                XPU void operator=(const float3 &a) const {
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

            decltype(float3::x) *LM_RESTRICT x;
            decltype(float3::y) *LM_RESTRICT y;
            decltype(float3::z) *LM_RESTRICT z;
            size_t size;
        };

#define LUMINOUS_SOA_BEGIN(StructName)  template<> \
struct SOA<StructName> {                           \
public:                                            \
SOA() = default;


#define LUMINOUS_SOA_MEMBER(StructName, MemberName) decltype(StructName::MemberName) *LM_RESTRICT MemberName;

#define LUMINOUS_SOA_MEMBERS(StructName, ...) MAP(LUMINOUS_SOA_MEMBER,StructName,__VA_ARGS__)

#define LUMINOUS_SOA_CONSTRUCTOR(...) \
SOA(size_t n, const std::shared_ptr<Device> &device) : size(n) { \
     \
    }

#define LUMINOUS_SOA_END size_t size; };

#define LUMINOUS_SOA(StructName, ...) LUMINOUS_SOA_BEGIN(StructName)\
        LUMINOUS_SOA_CONSTRUCTOR(__VA_ARGS__)                       \
        LUMINOUS_SOA_MEMBER(StructName,__VA_ARGS__)                                         \
        LUMINOUS_SOA_END \

        LUMINOUS_SOA(float2, x)

        void func() {
            int a, b;
            MAP(PRINT, a, b); /* Apply PRINT to a, b, and c */

            SOA<float3> soa(5, nullptr);
        }


    }
}