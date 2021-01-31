//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "common.h"

namespace luminous {
    namespace lstd {
        namespace span_internal {

            // Wrappers for access to container data pointers.
            template<typename C>
            XPU inline constexpr auto GetDataImpl(C &c, char) noexcept

                -> decltype(c.data()) {
                return c.

                data();
            }

            template<typename C>
            XPU inline constexpr auto GetData(C &c) noexcept

                -> decltype(GetDataImpl(c, 0)) {
                return
                GetDataImpl(c,
                0);
            }

            // Detection idioms for size() and data().
            template<typename C>
            using HasSize = std::is_integral<typename std::decay<decltype(std::declval<C &>().size())>::type>;

            // We want to enable conversion from vector<T*> to span<const T* const> but
            // disable conversion from vector<Derived> to span<Base>. Here we use
            // the fact that U** is convertible to Q* const* if and only if Q is the same
            // type or a more cv-qualified version of U.  We also decay the result type of
            // data() to avoid problems with classes which have a member function data()
            // which returns a reference.
            template<typename T, typename C>
            using HasData = std::is_convertible<
                        typename std::decay<decltype(GetData(std::declval<C &>()))>::type *, T *const *>;

        }  // namespace span_internal


    } // luminous::lstd
} // luminous