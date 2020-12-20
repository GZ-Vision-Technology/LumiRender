//
// Created by Zero on 2020/8/31.
//

#pragma once

#define CONSTEXPR constexpr

#define LUMINOUS_BEGIN namespace luminous {

#define LUMINOUS_END };

// From: https://github.com/Erlkoenig90/map-macro
#define LUMINOUS_MAP_EVAL0(...) __VA_ARGS__
#define LUMINOUS_MAP_EVAL1(...) LUMINOUS_MAP_EVAL0(LUMINOUS_MAP_EVAL0(LUMINOUS_MAP_EVAL0(__VA_ARGS__)))
#define LUMINOUS_MAP_EVAL2(...) LUMINOUS_MAP_EVAL1(LUMINOUS_MAP_EVAL1(LUMINOUS_MAP_EVAL1(__VA_ARGS__)))
#define LUMINOUS_MAP_EVAL3(...) LUMINOUS_MAP_EVAL2(LUMINOUS_MAP_EVAL2(LUMINOUS_MAP_EVAL2(__VA_ARGS__)))
#define LUMINOUS_MAP_EVAL4(...) LUMINOUS_MAP_EVAL3(LUMINOUS_MAP_EVAL3(LUMINOUS_MAP_EVAL3(__VA_ARGS__)))
#define LUMINOUS_MAP_EVAL5(...) LUMINOUS_MAP_EVAL4(LUMINOUS_MAP_EVAL4(LUMINOUS_MAP_EVAL4(__VA_ARGS__)))

#ifdef _MSC_VER
// MSVC needs more evaluations
#define LUMINOUS_MAP_EVAL6(...) LUMINOUS_MAP_EVAL5(LUMINOUS_MAP_EVAL5(LUMINOUS_MAP_EVAL5(__VA_ARGS__)))
#define LUMINOUS_MAP_EVAL(...)  LUMINOUS_MAP_EVAL6(LUMINOUS_MAP_EVAL6(__VA_ARGS__))
#else
#define LUMINOUS_MAP_EVAL(...)  LUMINOUS_MAP_EVAL5(__VA_ARGS__)
#endif

#define LUMINOUS_MAP_END(...)
#define LUMINOUS_MAP_OUT

#define LUMINOUS_MAP_EMPTY()
#define LUMINOUS_MAP_DEFER(id) id LUMINOUS_MAP_EMPTY()

#define LUMINOUS_MAP_GET_END2() 0, LUMINOUS_MAP_END
#define LUMINOUS_MAP_GET_END1(...) LUMINOUS_MAP_GET_END2
#define LUMINOUS_MAP_GET_END(...) LUMINOUS_MAP_GET_END1
#define LUMINOUS_MAP_NEXT0(test, next, ...) next LUMINOUS_MAP_OUT
#define LUMINOUS_MAP_NEXT1(test, next) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT0 ) ( test, next, 0)
#define LUMINOUS_MAP_NEXT(test, next)  LUMINOUS_MAP_NEXT1(LUMINOUS_MAP_GET_END test, next)

#define LUMINOUS_MAP0(f, x, peek, ...) f(x) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT(peek, LUMINOUS_MAP1) ) ( f, peek, __VA_ARGS__ )
#define LUMINOUS_MAP1(f, x, peek, ...) f(x) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT(peek, LUMINOUS_MAP0) ) ( f, peek, __VA_ARGS__ )

#define LUMINOUS_MAP_LIST0(f, x, peek, ...) , f(x) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT(peek, LUMINOUS_MAP_LIST1) ) ( f, peek, __VA_ARGS__ )
#define LUMINOUS_MAP_LIST1(f, x, peek, ...) , f(x) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT(peek, LUMINOUS_MAP_LIST0) ) ( f, peek, __VA_ARGS__ )
#define LUMINOUS_MAP_LIST2(f, x, peek, ...)   f(x) LUMINOUS_MAP_DEFER ( LUMINOUS_MAP_NEXT(peek, LUMINOUS_MAP_LIST1) ) ( f, peek, __VA_ARGS__ )

// Applies the function macro `f` to each of the remaining parameters.

#define LUMINOUS_MAP(f, ...) LUMINOUS_MAP_EVAL(LUMINOUS_MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Applies the function macro `f` to each of the remaining parameters and inserts commas between the results.

#define LUMINOUS_MAP_LIST(f, ...) LUMINOUS_MAP_EVAL(LUMINOUS_MAP_LIST2(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))


