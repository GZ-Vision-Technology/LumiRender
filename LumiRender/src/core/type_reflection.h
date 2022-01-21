//
// Created by Zero on 2021/9/16.
//

#pragma once

namespace luminous {
inline namespace reflection {

template<class Type>
struct runtime_class;// runtime class structure

template<class Tp, class... Bases>
class Meta_type_list {};// class hierarchy structure (used inside)

template<class Tp>
class Meta_inplace_type_t {};// used inside

// used for walking through class hierarchy structure for specific type
#define DECLARE_REFLECTION(this_class, ...)                        \
    friend struct luminous::reflection::runtime_class<this_class>; \
    using Meta_type = luminous::reflection::Meta_type_list<this_class, ##__VA_ARGS__>;

// Member mapping info
struct MemberMapEntry {
    const char name[63];// data member name
    bool skipped; // runtime mapping skipped indicator
    size_t field_offset;// offset from
};

#define LSTD_OFFSET_OF(s, m) \
    ((size_t) & reinterpret_cast<char const volatile &>(((s *) 0)->m))

#define LSTD_BASE_OFFSET_OF(s, b)                                                 \
    ((size_t) (reinterpret_cast<char *>(static_cast<b *>((s *) (uintptr_t) -1)) - \
               (char *) (uintptr_t) -1))

#define MEMBER_MAP_ENTRY(field) {#field, false, LSTD_OFFSET_OF(type, field)},
#define MEMBER_MAP_ENTRY_AUGMENTED(field, skipped) {#field, skipped, LSTD_OFFSET_OF(type, field)},

#define BEGIN_MEMBER_MAP(this_class)                                                    \
    const luminous::reflection::MemberMapEntry *this_class::_get_member_map_entries() { \
        using type = this_class;                                                        \
        static luminous::reflection::MemberMapEntry _member_map_entries[] = {

#define END_MEMBER_MAP()        \
    { 0, false, 0 }             \
    }                           \
    ;                           \
    return _member_map_entries; \
    }

#define DECLARE_MEMBER_MAP(this_class)                       \
    friend struct luminous::reflection::runtime_class<this_class>; \
    static const luminous::reflection::MemberMapEntry *_get_member_map_entries();

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
            if(!entry->skipped)
                fn(entry->name, base_offset + entry->field_offset);

        return;
    }

    template<class Fn, class Tp>
    static void _visit_this_member_map(const Tp *, size_t base_offset,
                                       const Fn &fn) {}

    template<class Fn, class Tp, class... Bases>
    static void _visit_bases_member_map(const Meta_type_list<Tp, Bases...> *,
                                        size_t base_offset, const Fn &fn) {
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

}// namespace reflection
}// namespace luminous