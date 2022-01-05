//
// Created by Zero on 2021/9/16.
//

#pragma once

#include <iostream>

namespace luminous {
    inline namespace refl {

    template<class Type>
    struct runtime_class;// runtime class structure

    template<class Tp, class... Bases>
    class Meta_type_list {};// class hierarchy structure (used inside)

    template<class Tp>
    class Meta_inplace_type_t {};// used inside

    // used for walking through class hierarchy structure for specific type
#define DECLARE_REFLECTION(this_class, ...)        \
    friend struct luminous::refl::runtime_class<this_class>; \
    using Meta_type = luminous::refl::Meta_type_list<this_class, ##__VA_ARGS__>;

    // Member mapping info
    struct MemberMapEntry {
        const char name[84];// data member name
        size_t field_offset;// offset from
    };

#define LSTD_OFFSET_OF(s, m) \
    ((size_t) & reinterpret_cast<char const volatile &>(((s *) 0)->m))

#define LSTD_BASE_OFFSET_OF(s, b)                                                 \
    ((size_t) (reinterpret_cast<char *>(static_cast<b *>((s *) (uintptr_t) -1)) - \
               (char *) (uintptr_t) -1))

#define MEMBER_MAP_ENTRY(field) {#field, LSTD_OFFSET_OF(type, field)},

#define BEGIN_MEMBER_MAP(this_class)                                              \
    const luminous::refl::MemberMapEntry *this_class::_get_member_map_entries() { \
        using type = this_class;                                                  \
        static luminous::refl::MemberMapEntry _member_map_entries[] = {

#define END_MEMBER_MAP()        \
    { 0, 0 }                    \
    }                           \
    ;                           \
    return _member_map_entries; \
    }

#define DECLARE_MEMBER_MAP(this_class)                       \
    friend struct luminous::refl::runtime_class<this_class>; \
    static const luminous::refl::MemberMapEntry *_get_member_map_entries();

    template<class Type>
    struct runtime_class {
    private:
        // SFINAE type identification
        template<class Tp>
        static const typename Tp::Meta_type *
        _get_meta_type(const typename Tp::Meta_type *);// has _Meta_type inside
        template<class Tp>
        static const Tp *_get_meta_type(const Tp *);// no _Meta_type inside

        // SFINAE type identification
        template<class Tp>
        static auto
        _get_member_map_entries(const decltype(Tp::_get_member_map_entries) *)
                -> const
                Meta_inplace_type_t<Tp> *;// has _get_member_map_entries class member
        template<class Tp>
        static auto _get_member_map_entries(const Tp *)
                -> const Tp *;// no _get_member_map_entries

        template<class Tp>
        using Runtime_class_type_t =
                decltype(_get_meta_type<Tp>(nullptr));// generic type for meta type

        template<class Tp>
        using Member_map_t = decltype(_get_member_map_entries<Tp>(
                nullptr));// generic type for member mapping

        template<class Fn, class Tp>
        static void _visit_this_member_map(const Meta_inplace_type_t<Tp> *,
                                           size_t base_offset, const Fn &fn) {

            auto entry = Tp::_get_member_map_entries();

            for (; entry != nullptr && entry->name[0] != 0; ++entry)
                fn(entry->name, base_offset + entry->field_offset);

            return;
        }

        template<class Fn, class Tp>
        static void _visit_this_member_map(const Tp *, size_t base_offset,
                                           const Fn &fn) {}

        template<class Fn, class Tp, class... Bases>
        static void _visit_bases_member_map(const Meta_type_list<Tp, Bases...> *,
                                            size_t base_offset, const Fn &fn) {
            ((std::cout << "offset: " << LSTD_BASE_OFFSET_OF(Tp, Bases) << std::endl),
             ...);
            (..., (runtime_class<Bases>::visit_member_map(
                          base_offset + LSTD_BASE_OFFSET_OF(Tp, Bases), fn)));
        }

        template<class Fn, class Tp, class... Bases>
        static void _visit_bases_member_map(const Tp *, size_t base_offset,
                                            const Fn &fn) {}

    public:
        template<class Fn>
        static void visit_member_map(size_t base_offset, const Fn &fn) {
            _visit_bases_member_map(Runtime_class_type_t<Type>{}, base_offset, fn);
            _visit_this_member_map(Member_map_t<Type>{}, base_offset, fn);
        }
    };

#define REFL_MAX_MEMBER_COUNT 128

        template<int N>
        struct Int : Int<N - 1> {
        };

        template<>
        struct Int<0> {
        };

        namespace detail {
            template<int N>
            struct Sizer {
                char _[N];
            };
        }

#define REFL_CLASS(NAME)                                                        \
    using ReflSelf = NAME;                                                      \
    static refl::detail::Sizer<1> _member_counter(int, ...);                    \
    template<int N>                                                             \
    struct MemberRegister {                                                     \
        template<typename F>                                                    \
        static void process(const F &f) {}                                      \
    };

#define DEFINE_AND_REGISTER_MEMBER(TYPE, NAME, ...)                             \
    TYPE NAME{__VA_ARGS__};                                                     \
    static constexpr int NAME##_refl_index =                                    \
        sizeof((_member_counter(0,                                              \
            (refl::Int<REFL_MAX_MEMBER_COUNT>*)nullptr)));                      \
    static_assert(NAME##_refl_index <= REFL_MAX_MEMBER_COUNT,                   \
                "index must not greater than REFL_MAX_MEMBER_COUNT");           \
    static refl::detail::Sizer<NAME##_refl_index + 1>                           \
        (_member_counter(int, refl::Int<NAME##_refl_index + 1> *));             \
    template<>                                                                  \
    struct MemberRegister<NAME##_refl_index - 1> {                              \
        template<typename F>                                                    \
        static void process(const F &f) {                                       \
            f(&ReflSelf::NAME, #NAME);                                          \
        }                                                                       \
    };

        namespace detail {
            template<typename T, typename F, int...Is>
            void for_each_registered_member_aux(const F &f, std::integer_sequence<int, Is...>) {
                (T::template MemberRegister<Is>::template process<F>(f), ...);
            }

            template<typename T, typename F>
            void for_each_registered_member(const F &f) {
                for_each_registered_member_aux<T>(
                        f, std::make_integer_sequence<int, REFL_MAX_MEMBER_COUNT>());
            }

        }

        template<typename T, typename F>
        void for_each_registered_member(const F &f) {
#define OFFSET_OF(Class, member) reinterpret_cast<size_t>(&((*(Class *) 0).*member))
            detail::for_each_registered_member<T>([&](auto member_ptr, const char *name) {
                f(OFFSET_OF(T, member_ptr), name);
            });
#undef OFFSET_OF
        }

        template<typename...T>
        struct BaseBinder : public T ... {
        public:
            using Bases = std::tuple<T...>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            static_assert(base_num == 1, "Temporarily disable multiple inheritance");

            template<int idx>
            using Base = std::tuple_element_t<idx, Bases>;

            BaseBinder() = default;

            explicit BaseBinder(T &&...args)
                    : T{std::move(args)}... {}

        };

        template<>
        struct BaseBinder<> {
        public:
            using Bases = std::tuple<>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            template<int idx>
            using Base = void;

            BaseBinder() = default;

        };

        template<typename T>
        struct BaseBinder<T> : public T {
        public:
            using Bases = std::tuple<T>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            template<int idx>
            using Base = T;

            BaseBinder() = default;

            using T::T;

            explicit BaseBinder(T &&t)
                    : T{std::move(t)} {}

        };

#define BASE_CLASS(...) public BaseBinder<__VA_ARGS__>

#define RENDER_CLASS_HEAD(ClassName) using BaseBinder::BaseBinder; \
                                          GEN_BASE_NAME(ClassName) \
                                        REFL_CLASS(ClassName)

        namespace detail {
            template<typename T, typename F, int...Is>
            void for_each_direct_base_aux(const F &f, std::integer_sequence<int, Is...>) {
                (f.template operator()<std::tuple_element_t<Is, typename T::Bases>>(), ...);
            }
        }

        template<typename T, typename F>
        void for_each_direct_base(const F &f) {
            detail::for_each_direct_base_aux<T>(
                    f, std::make_integer_sequence<int, std::tuple_size_v<typename T::Bases>>());
        }

        template<typename F>
        struct Visitor {
            F func;

            explicit Visitor(const F &f) : func(f) {}

            template<typename T>
            void operator()(T *ptr = nullptr) const {
                for_each_direct_base<T>(*this);
                func((T *) nullptr);
            }
        };

        template<typename T, typename F>
        void for_each_all_base(const F &f) {
            Visitor <F> visitor(f);
            for_each_direct_base<T>(visitor);
        }

        template<typename T, typename F>
        void for_each_all_registered_member(const F &func) {
            for_each_all_base<T>([&](auto ptr) {
                using Base = std::remove_pointer_t<decltype(ptr)>;
                for_each_registered_member<Base>([&](auto offset, auto member_name) {
                    func(offset, member_name, ptr);
                });
            });
            for_each_registered_member<T>([&](auto offset, auto member_name) {
                func(offset, member_name, (T *) nullptr);
            });
        }
    }
}