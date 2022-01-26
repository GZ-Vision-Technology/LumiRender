//
// Created by Zero on 2021/2/4.
//


#pragma once


#include <typeinfo>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
    #ifdef FOUND_VLD
    #include <vld.h>
    #endif
    #include "crtdbg.h"
#endif

// _countof definination
#ifndef _countof
extern "C++"
{
    template <typename _CountofType, size_t _SizeOfArray>
    char (*__countof_helper(_CountofType (&_Array)[_SizeOfArray]))[_SizeOfArray];

    #define __crt_countof(_Array) (sizeof(*__countof_helper(_Array)) + 0)
}
#define _countof __crt_countof
#endif


#if defined(__GNUC__) || defined(__clang__)
#define LM_FUNCSIG ""
#elif defined(_MSC_VER)
#define LM_FUNCSIG __FUNCSIG__
#endif

#ifndef LM_FUNCSIG
#error "LM_FUNCSIG is needed but not provided by C++ compiler."
#endif

#if defined(_MSC_VER)
#define HAVE_ALIGNED_MALLOC
#endif

#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif

#define F_INLINE __forceinline

#define INLINE inline

#define LM_XPU_INLINE LM_XPU INLINE

#define LM_GPU_INLINE LM_GPU INLINE

#define LM_RESTRICT __restrict
#define LM_NODISCARD [[nodiscard]]
#define LM_ND_XPU LM_NODISCARD LM_XPU
#define LM_ND_INLINE LM_NODISCARD INLINE
#define ND_XPU_INLINE LM_NODISCARD LM_XPU_INLINE

#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x

#if defined(__CUDACC__)
#define IS_GPU_CODE
#endif

#define HAVE_POSIX_MEMALIGN

#ifdef IS_GPU_CODE

#define lm_likely(x) __builtin_expect(!!(x), 1)
#define lm_unlikely(x) __builtin_expect(!!(x), 0)

#define LUMINOUS_DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#define LM_XPU __host__ __device__
#define LM_GPU __device__
// host/device constant data visibility modifier
#define LM_XPU_CONSTANT_VISIBILITY __constant__

#define CPU_ONLY(...)

#define GEN_NAME_FUNC LM_ND_XPU const char *name() {             \
                                        LUMINOUS_VAR_DISPATCH(name);\
                                   }

#define GEN_STRING_FUNC(args)
#else

#define lm_likely(x) (x)
#define lm_unlikely(x) (x)

#define LM_XPU
#define LM_GPU
// host/device constant data visibility modifier
#define LM_XPU_CONSTANT_VISIBILITY
#define LUMINOUS_DBG(...) fprintf(stderr, __FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)

#define CPU_ONLY(...) __VA_ARGS__

#define GEN_STRING_FUNC(args) LM_NODISCARD std::string to_string() const args

#endif

#define LUMINOUS_TO_STRING(...) return string_printf(__VA_ARGS__);

#ifndef NDEBUG

#define DEBUG_ONLY(...) __VA_ARGS__

#else

#define DEBUG_ONLY(...)

#endif

#define GEN_TO_STRING_FUNC GEN_STRING_FUNC({LUMINOUS_VAR_DISPATCH(to_string);})

#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#define USE_LSTD 1


#ifdef NDEBUG
#define EXE_DEBUG(pred, arg)
#define LM_ASSERT(condition, ...)
#else
#define EXE_DEBUG(pred, arg) if (pred) { arg; }

#define LM_ASSERT(condition, ...)                      \
        [&] {                                          \
             bool b = (condition);                     \
             if (!b) { printf(__VA_ARGS__);} assert(b);\
        }()

#endif



#define DCHECK(a) assert(a);
#define DCHECK_EQ(a, b) DCHECK((a) == (b))
#define DCHECK_GT(a, b) DCHECK((a) > (b));
#define DCHECK_GE(a, b) DCHECK((a) >= (b));
#define DCHECK_LT(a, b) DCHECK((a) < (b));
#define DCHECK_LE(a, b) DCHECK((a) <= (b));

#define CONTINUE_IF(condition) if((condition)) { continue; }
#define CONTINUE_IF_TIPS(condition, str) if((condition)) { LUMINOUS_DBG(str); continue; }

#define BREAK_IF(condition) if((condition)) { break; }
#define BREAK_IF_TIPS(condition, str) if((condition)) { LUMINOUS_DBG(str); break; }

namespace luminous {
    using ptr_t = uint64_t;
}

#ifdef _MSC_VER

#define _DECL_ETW_ALLOCATOR      __declspec(allocator)

#endif

#ifdef __GNUC__

#define _DECL_ETW_ALLOCATOR

#define _aligned_malloc(size, alignment)   ::aligned_alloc(alignment, size)
#define _aligned_free(p)                   ::free(p)

#define _CRT_GUARDOVERFLOW

#endif // __GNUC__
