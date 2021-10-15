//
// Created by Zero on 15/10/2021.
//


#pragma once

#include <iostream>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>

namespace luminous {
    inline namespace lstd {
        template<class T>
        struct remove_cvref {
            using type = std::remove_cv_t<std::remove_reference_t<T>>;
        };

        template<typename T>
        using remove_cvref_t = typename remove_cvref<T>::type;

        namespace detail {
            template<typename F>
            struct FunctionTraitImpl {

            };

            template<typename Ret, typename...Args>
            struct FunctionTraitImpl<std::function<Ret(Args...)>> {
                using R = Ret;
                using As = std::tuple<Args...>;
            };
        }

        template<typename F>
        struct FunctionTrait {
        private:
            using Impl = detail::FunctionTraitImpl<decltype(std::function{std::declval<remove_cvref_t<F>>()})>;
        public:
            using Ret = typename Impl::R;
            using Args = typename Impl::As;
        };
    }
}