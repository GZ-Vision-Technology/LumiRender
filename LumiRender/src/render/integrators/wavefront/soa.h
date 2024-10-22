//
// Created by Zero on 27/08/2021.
//


#pragma once

#include <type_traits>
#include "gpu/framework/helper/cuda.h"
#include "core/macro_map.h"

namespace luminous {
    inline namespace render {
        template<typename T>
        struct SOA {
            static constexpr bool definitional = false;
        };
#ifndef __CUDACC__

        template<typename T, typename TDevice>
        struct SOAMember {

            static auto create(int n, const TDevice &device) {
                if constexpr (SOA<T>::definitional) {
                    return SOA<T>(n, device);
                } else {
                    return device->template obtain_restrict_ptr<T>(n);
                }
            }

            template<typename TParam>
            static auto clone_to_host(TParam param, int n, TDevice device) {
                if constexpr(!std::is_pointer_v<TParam>) {
                    return param.to_host(device);
                } else {
                    TParam p = device->template obtain_restrict_ptr<std::remove_pointer_t<TParam>>(n);
                    download(p, reinterpret_cast<ptr_t>(param), n);
                    return p;
                }
            }

            using type = decltype(create(0, nullptr));
        };

#else
        template<typename T>
        struct SOAMember {
            static auto create() {
                if constexpr (SOA<T>::definitional) {
                    return SOA<T>();
                } else {
                    return static_cast<T *>(nullptr);
                }
            }

            using type = decltype(create());
        };
#endif

    }
}

#define LUMINOUS_SOA_BEGIN(StructName)  template<> \
struct SOA<StructName> {                           \
public:                                            \
static constexpr bool definitional = true;         \
using element_type = StructName;                   \
SOA() = default;                                   \
int capacity;

#ifndef __CUDACC__
    // member definition
    #define LUMINOUS_SOA_MEMBER(MemberName) SOAMember<decltype(element_type::MemberName),Device*>::type MemberName;
#else
    #define LUMINOUS_SOA_MEMBER(MemberName) SOAMember<decltype(element_type::MemberName)>::type MemberName;
#endif

#define LUMINOUS_SOA_MEMBERS(...) MAP(LUMINOUS_SOA_MEMBER,__VA_ARGS__)

#ifndef __CUDACC__
    // constructor definition
    #define LUMINOUS_SOA_MEMBER_ASSIGNMENT(MemberName) MemberName =          \
    SOAMember<decltype(element_type::MemberName),Device*>::create(n, device);
    #define LUMINOUS_SOA_CONSTRUCTOR(...)                                    \
    SOA(int n, Device *device) : capacity(n) {                               \
    MAP(LUMINOUS_SOA_MEMBER_ASSIGNMENT,__VA_ARGS__) }

#else
    #define LUMINOUS_SOA_CONSTRUCTOR(...)
#endif


// assignment function definition
#define LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT(MemberName) this->MemberName = s.MemberName;
#define LUMINOUS_SOA_ASSIGNMENT(...)                               \
LM_XPU SOA &operator=(const SOA &s) { capacity = s.capacity;       \
MAP(LUMINOUS_SOA_ASSIGNMENT_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__)    \
return *this; }

// clone to host function definition
#define LUMINOUS_SOA_TO_HOST_BODY_MEMBER_CLONE(MemberName) \
ret.MemberName = SOAMember<decltype(element_type::MemberName), TDevice*>::clone_to_host(MemberName, capacity, device);
#define LUMINOUS_SOA_TO_HOST(...)                               \
template<typename TDevice>                                      \
SOA<element_type> to_host(TDevice * device) const {             \
DCHECK(device->is_cpu())                                        \
auto ret = SOA<element_type>(capacity, device);                 \
MAP(LUMINOUS_SOA_TO_HOST_BODY_MEMBER_CLONE,__VA_ARGS__)         \
return ret; }

// access function definition
#define LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT(MemberName) r.MemberName = this->MemberName[i];
#define LUMINOUS_SOA_ACCESSOR(...)                                                     \
LM_XPU element_type operator[](int i) const { DCHECK_LT(i, capacity); element_type r;  \
MAP(LUMINOUS_SOA_ACCESSOR_BODY_MEMBER_ASSIGNMENT,__VA_ARGS__) return r; }

// get set struct
#define LUMINOUS_SOA_GET_SET_CASTER_IMPL(MemberName) r.MemberName = soa->MemberName[i];
#define LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL(MemberName) soa->MemberName[i] = a.MemberName;
#define LUMINOUS_SOA_SET_GET_STRUCT(...) struct GetSetIndirector { SOA *soa;int i;    \
LM_XPU operator element_type() const { element_type r;                                \
MAP(LUMINOUS_SOA_GET_SET_CASTER_IMPL, __VA_ARGS__) return r;}                         \
LM_XPU void operator=(const element_type &a) const {                                  \
MAP(LUMINOUS_SOA_GET_SET_ASSIGNMENT_IMPL, __VA_ARGS__) }};

// get set accessor
#define LUMINOUS_SOA_INDIRECTOR_ACCESSOR LM_XPU GetSetIndirector operator[](int i) { \
DCHECK_LT(i, capacity);                                                              \
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
        LUMINOUS_SOA_TO_HOST(__VA_ARGS__)                           \
        LUMINOUS_SOA_END


