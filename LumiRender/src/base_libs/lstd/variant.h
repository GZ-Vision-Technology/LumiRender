//
// Created by Zero on 2021/1/27.
//


#pragma once

#include "../header.h"
#include "common.h"


namespace lstd {

    namespace detail {
        template<typename... T>
        struct TypeIndex {
            template<typename U, typename Tp, typename... Rest>
            struct GetIndex_ {
                static const int value =
                        std::is_same<Tp, U>::value
                        ? 0
                        : ((GetIndex_<U, Rest...>::value == -1) ? -1 : 1 + GetIndex_<U, Rest...>::value);
            };
            template<typename U, typename Tp>
            struct GetIndex_<U, Tp> {
                static const int value = std::is_same<Tp, U>::value ? 0 : -1;
            };
            template<int I, typename Tp, typename... Rest>
            struct GetType_ {
                using type = typename std::conditional<I == 0, Tp, typename GetType_<I - 1, Rest...>::type>::type;
            };

            template<int I, typename Tp>
            struct GetType_<I, Tp> {
                using type = typename std::conditional<I == 0, Tp, void>::type;
            };

            template<typename U>
            struct GetIndex {
                static const int value = GetIndex_<U, T...>::value;
            };

            template<int N>
            struct GetType {
                using type = typename GetType_<N, T...>::type;
            };
        };
        template<class T, class... Rest>
        struct FirstOf {
            using type = T;
        };

        template<typename U, typename... T>
        struct SizeOf {
            static constexpr int value = std::max<int>(sizeof(U), SizeOf<T...>::value);
        };
        template<typename T>
        struct SizeOf<T> {
            static constexpr int value = sizeof(T);
        };
    }
    using namespace detail;

    template<typename... T>
    struct Variant {
    private:

        static constexpr int nTypes = sizeof...(T);
        static constexpr std::size_t alignment_value = std::max({alignof(T)...});
        typename std::aligned_storage<SizeOf<T...>::value, alignment_value>::type data;
        int index = -1;

    public:
        using TypeTuple = std::tuple<T...>;
        using Index = TypeIndex<T...>;
        using FirstType = typename FirstOf<T...>::type;
        static constexpr size_t num_types = nTypes;

        Variant() = default;

        template<typename U>
        XPU explicit Variant(const U &u) {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new(&data) U(u);
            index = Index::template GetIndex<U>::value;
        }

        XPU Variant(const Variant &v) : index(v.index) {
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new(&data) U(item);
            });
        }

        XPU [[nodiscard]] int type_index() const { return index; }

        template<typename U>
        XPU constexpr static int index_of() {
            return Index::template GetIndex<U>::value;
        }

        XPU Variant &operator=(const Variant &v) noexcept {
            if (this == &v)
                return *this;
            if (index != -1)
                _drop();
            index = v.index;
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new(&data) U(item);
            });
            return *this;
        }

        XPU Variant(Variant &&v) noexcept: index(v.index) {
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
        }

        XPU Variant &operator=(Variant &&v) noexcept {
            if (index != -1)
                _drop();
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
            return *this;
        }

        template<typename U>
        XPU Variant &operator=(const U &u) {
            if (index != -1) {
                _drop();
            }
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new(&data) U(u);
            index = Index::template GetIndex<U>::value;
            return *this;
        }

        NDSC XPU bool null() const { return index == -1; }

        template<typename U>
        NDSC XPU bool isa() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value == index;
        }

        template<typename U>
        NDSC XPU U *get() {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<U *>(&data);
        }

        template<typename U>
        NDSC XPU const U *get() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<const U *>(&data);
        }

#define _GEN_CASE_N(N)                                                                                                 \
    case N:                                                                                                            \
        if constexpr (N < nTypes) {                                                                                    \
            using ty = typename Index::template GetType<N>::type;                                                      \
            if constexpr (!std::is_same_v<ty, std::monostate>) {                                                       \
                if constexpr (std::is_const_v<std::remove_pointer_t<decltype(this)>>)                                  \
                    return visitor(*reinterpret_cast<const ty *>(&data));                                              \
                else                                                                                                   \
                    return visitor(*reinterpret_cast<ty *>(&data));                                                    \
            }                                                                                                          \
        };                                                                                                             \
        break;
#define _GEN_CASES_2()                                                                                                 \
    _GEN_CASE_N(0)                                                                                                     \
    _GEN_CASE_N(1)
#define _GEN_CASES_4()                                                                                                 \
    _GEN_CASES_2()                                                                                                     \
    _GEN_CASE_N(2)                                                                                                     \
    _GEN_CASE_N(3)
#define _GEN_CASES_8()                                                                                                 \
    _GEN_CASES_4()                                                                                                     \
    _GEN_CASE_N(4)                                                                                                     \
    _GEN_CASE_N(5)                                                                                                     \
    _GEN_CASE_N(6)                                                                                                     \
    _GEN_CASE_N(7)
#define _GEN_CASES_16()                                                                                                \
    _GEN_CASES_8()                                                                                                     \
    _GEN_CASE_N(8)                                                                                                     \
    _GEN_CASE_N(9)                                                                                                     \
    _GEN_CASE_N(10)                                                                                                    \
    _GEN_CASE_N(11)                                                                                                    \
    _GEN_CASE_N(12)                                                                                                    \
    _GEN_CASE_N(13)                                                                                                    \
    _GEN_CASE_N(14)                                                                                                    \
    _GEN_CASE_N(15)
#define _GEN_DISPATCH_BODY()                                                                                           \
    using Ret = std::invoke_result_t<Visitor, typename FirstOf<T...>::type &>;                                         \
    static_assert(nTypes <= 16, "too many types");                                                                     \
    if constexpr (nTypes <= 2) {                                                                                       \
        switch (index) { _GEN_CASES_2(); }                                                                             \
    } else if constexpr (nTypes <= 4) {                                                                                \
        switch (index) { _GEN_CASES_4(); }                                                                             \
    } else if constexpr (nTypes <= 8) {                                                                                \
        switch (index) { _GEN_CASES_8(); }                                                                             \
    } else if constexpr (nTypes <= 16) {                                                                               \
        switch (index) { _GEN_CASES_16(); }                                                                            \
    } else  {                                                                                                          \
        assert(0);                                                                                                     \
    }                                                                                                                  \
    if constexpr (std::is_same_v<void, Ret>) {                                                                         \
        return;                                                                                                        \
    } else {                                                                                                           \
        assert(0);                                                                                                     \
    }

        template<class Visitor>
        XPU decltype(auto) dispatch(Visitor &&visitor) {
            _GEN_DISPATCH_BODY()
        }

        template<class Visitor>
        XPU decltype(auto) dispatch(Visitor &&visitor) const {
            _GEN_DISPATCH_BODY()
        }

        XPU ~Variant() {
            if (index != -1)
                _drop();
        }

    private:
        XPU void _drop() {
            auto *that = this; // prevent gcc ICE
            dispatch([=](auto &&self) {
                using U = std::decay_t<decltype(self)>;
                that->template get<U>()->~U();
            });
        }

#undef _GEN_CASE_N

#define LUMINOUS_VAR_DISPATCH(method, ...)                                                                             \
    return this->dispatch([&, this](auto &&self)-> decltype(auto) {                                                    \
        return self.method(__VA_ARGS__);                                                                               \
    });
#define LUMINOUS_VAR_PTR_DISPATCH(method, ...)                                                                         \
    return this->dispatch([&, this](auto &&self)-> decltype(auto) {                                                    \
        return self->method(__VA_ARGS__);                                                                              \
    });

    };


}; // lstd
