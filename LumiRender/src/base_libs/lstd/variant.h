//
// Created by Zero on 2021/1/27.
//


#pragma once

#include "../header.h"
#include "common.h"
#include "core/type_reflection.h"
#include <algorithm>

namespace luminous {
    inline namespace lstd {

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

        template<typename... Ts>
        struct Variant {
        public:

            static constexpr bool is_pointer_type = ((std::is_pointer_v<Ts>) && ...);

            static_assert(((!std::is_pointer_v<std::remove_pointer_t<Ts>>) && ...),
                          "Ts can not be the secondary pointer!");
            static_assert((is_pointer_type) || ((!std::is_pointer_v<Ts>) && ...),
                          "Ts must be consistency!");

            // size_for_ptr use for pointer type
            static constexpr std::size_t size_for_ptr() {
                static_assert(is_pointer_type, "size_for_ptr() must be use for pointer type!");
                return std::max({sizeof(std::remove_pointer_t<Ts>)...});
            }

            // size_for_ptr use for pointer type
            static constexpr std::size_t alignment_for_ptr() {
                static_assert(is_pointer_type, "alignment_for_ptr() must be use for pointer type!");
                return std::max({alignof(std::remove_pointer_t<Ts>)...});
            }

        protected:
            static constexpr int nTypes = sizeof...(Ts);

            static constexpr std::size_t alignment_value = std::max({alignof(Ts)...});

            typename std::aligned_storage<std::max({sizeof(Ts)...}), alignment_value>::type data{};

            int8_t index = -1;

            DECLARE_MEMBER_MAP(Variant)

        public:
            using TypeTuple = std::tuple<Ts...>;
            using Index = TypeIndex<Ts...>;
            using FirstType = typename FirstOf<Ts...>::type;
            static constexpr size_t num_types = nTypes;

            Variant() = default;

            template<typename U>
            LM_XPU explicit Variant(const U &u) {
                static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
                new(&data) U(u);
                index = Index::template GetIndex<U>::value;
            }

            LM_XPU Variant(const Variant &v) : index(v.index) {
              if(index >= 0) {
                v.dispatch([&](const auto &item) {
                    using U = std::decay_t<decltype(item)>;
                    new(&data) U(item);
                });
            }
            }

            LM_ND_XPU int type_index() const { return index; }

            template<typename U>
            LM_XPU constexpr static int index_of() {
                return Index::template GetIndex<U>::value;
            }

            LM_XPU Variant &operator=(const Variant &v) noexcept {
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

            LM_XPU Variant(Variant &&v) noexcept: index(v.index) {
                index = v.index;
                v.index = -1;
                std::memcpy(&data, &v.data, sizeof(data));
            }

            LM_XPU Variant &operator=(Variant &&v) noexcept {
                if (index != -1)
                    _drop();
                index = v.index;
                v.index = -1;
                std::memcpy(&data, &v.data, sizeof(data));
                return *this;
            }

            template<typename U>
            LM_XPU Variant &operator=(const U &u) {
                if (index != -1) {
                    _drop();
                }
                static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
                new(&data) U(u);
                index = Index::template GetIndex<U>::value;
                return *this;
            }

            LM_NODISCARD LM_XPU bool null() const { return index == -1; }

            template<typename U>
            LM_NODISCARD LM_XPU bool isa() const {
                static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
                return Index::template GetIndex<U>::value == index;
            }

            // use of prototype
            template<typename U>
            LM_NODISCARD LM_XPU U *get() {
                static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
                return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<U *>(&data);
            }

            template<typename U>
            LM_NODISCARD LM_XPU const U *get() const {
                static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
                return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<const U *>(&data);
            }

            // use for pointer type
            template<typename U>
            LM_ND_XPU const U *as() const {
                static_assert(is_pointer_type, "as<U>() use for pointer type!");
                static_assert(Index::template GetIndex<U *>::value != -1, "U* is not in T...");
                return *get<U *>();
            }

            template<typename U>
            LM_ND_XPU U *as() {
                static_assert(is_pointer_type, "as<U>() use for pointer type!");
                static_assert(Index::template GetIndex<U *>::value != -1, "U* is not in T...");
                return *get<U *>();
            }

#define GEN_CASE_N(N)                                                                                                     \
        case N:                                                                                                            \
            if constexpr ((N) < nTypes) {                                                                                    \
                using ty = typename Index::template GetType<N>::type;                                                      \
                if constexpr (!std::is_same_v<ty, std::monostate>) {                                                       \
                    if constexpr (std::is_const_v<std::remove_pointer_t<decltype(this)>>)                                  \
                        return visitor(*reinterpret_cast<const ty *>(&data));                                              \
                    else                                                                                                   \
                        return visitor(*reinterpret_cast<ty *>(&data));                                                    \
                }                                                                                                          \
            }                                                                                                             \
            break;
#define GEN_CASES_2()                                                                                                     \
        GEN_CASE_N(0)                                                                                                     \
        GEN_CASE_N(1)
#define GEN_CASES_4()                                                                                                     \
        GEN_CASES_2()                                                                                                     \
        GEN_CASE_N(2)                                                                                                     \
        GEN_CASE_N(3)
#define GEN_CASES_8()                                                                                                     \
        GEN_CASES_4()                                                                                                     \
        GEN_CASE_N(4)                                                                                                     \
        GEN_CASE_N(5)                                                                                                     \
        GEN_CASE_N(6)                                                                                                     \
        GEN_CASE_N(7)
#define GEN_CASES_16()                                                                                                    \
        GEN_CASES_8()                                                                                                     \
        GEN_CASE_N(8)                                                                                                     \
        GEN_CASE_N(9)                                                                                                     \
        GEN_CASE_N(10)                                                                                                    \
        GEN_CASE_N(11)                                                                                                    \
        GEN_CASE_N(12)                                                                                                    \
        GEN_CASE_N(13)                                                                                                    \
        GEN_CASE_N(14)                                                                                                    \
        GEN_CASE_N(15)
#define GEN_DISPATCH_BODY()                                                                        \
  using Ret = std::invoke_result_t<Visitor, typename FirstOf<Ts...>::type &>;                      \
  static_assert(nTypes <= 16, "too many types");                                                   \
  /*LM_ASSERT(index < 0, "Error: %s: unknown type tag: %d\n", func_name, index);  */               \
                                                                                                   \
  if constexpr (nTypes <= 2) {                                                                     \
    switch (index) { GEN_CASES_2() }                                                               \
  } else if constexpr (nTypes <= 4) {                                                              \
    switch (index) { GEN_CASES_4() }                                                               \
  } else if constexpr (nTypes <= 8) {                                                              \
    switch (index) { GEN_CASES_8() }                                                               \
  } else if constexpr (nTypes <= 16) {                                                             \
    switch (index) { GEN_CASES_16() }                                                              \
  } else {                                                                                         \
    LM_ASSERT(0, "Error: %s: unknown type tag: %d\n", func_name, index);                           \
  }                                                                                                \
  if constexpr (std::is_same_v<void, Ret>) {                                                       \
    return;                                                                                        \
  } else {                                                                                         \
          /*  LM_ASSERT(0, "Error: %s: unknown type tag: %d\n", func_name, index);*/               \
    return Ret{};                                                                                  \
        }

            template<class Visitor>
            LM_XPU decltype(auto) dispatch(Visitor &&visitor, const char *func_name = nullptr) {
                GEN_DISPATCH_BODY()
            }

            template<class Visitor>
            LM_XPU decltype(auto) dispatch(Visitor &&visitor, const char *func_name = nullptr) const {
                GEN_DISPATCH_BODY()
            }

            LM_XPU ~Variant() {
                if (index != -1)
                    _drop();
            }

        private:
            LM_XPU void _drop() {
                auto *that = this; // prevent gcc ICE
                dispatch([=](auto &&self) {
                    using U = std::decay_t<decltype(self)>;
                    that->template get<U>()->~U();
                });
                index = -1;
            }

#undef _GEN_CASE_N

#define LUMINOUS_VAR_DISPATCH(method, ...)                                                                                 \
        return this->dispatch([&](auto &&self)-> decltype(auto) {                                                          \
            return self.method(__VA_ARGS__);                                                                               \
        }, "File:\"" __FILE__ "\",Line:" TO_STRING(__LINE__) ",Calling:" LM_FUNCSIG);
#define LUMINOUS_VAR_PTR_DISPATCH(method, ...)                                                                             \
        return this->dispatch([&](auto &&self)-> decltype(auto) {                                                          \
            return self->method(__VA_ARGS__);                                                                              \
        }, "File:\"" __FILE__ "\",Line:" TO_STRING(__LINE__) ",Calling:" LM_FUNCSIG);

        };

        template<class ...Types> inline
        BEGIN_MEMBER_MAP(Variant<Types...>)
            MEMBER_MAP_ENTRY_AUGMENTED(data, !is_pointer_type)
        END_MEMBER_MAP()

        template<typename T>
        class Sizer {
        public:
            static constexpr size_t compound_size() {
                return T::size_for_ptr() + T::alignment_for_ptr()
                       + sizeof(T) + alignof(T);
            }

            static constexpr size_t size = sizeof(T) + alignof(T);
        };

    }; // lstd
}
