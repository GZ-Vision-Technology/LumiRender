//
// Created by Zero on 16/10/2021.
//


#pragma once

#include "cpu/cpu_impl.h"
#include "gpu/framework/cuda_impl.h"

namespace luminous {
    inline namespace render {
        template<typename T>
        class TKernel {
        public:
            using func_type = T;
            using function_trait = FunctionTrait<func_type>;
        protected:
            func_type _func{};
            uint64_t _handle{};
        public:

            explicit TKernel(func_type func) : _func(func) {}

            void configure(uint3 grid_size,
                           uint3 local_size,
                           size_t sm) {}

                           template<typename Ret, typename...Args, size_t...Is>
                           void call_impl(Ret(*f)(Args...), void *args[], std::index_sequence<Is...>) {
                f(*static_cast<std::tuple_element_t<Is, std::tuple<Args...>> *>(args[Is])...);
            }

            template<typename Ret, typename...Args>
            void call(Ret(*f)(Args...), void *args[]) {
                call_impl(f, args, std::make_index_sequence<sizeof...(Args)>());
            }

            template<typename...Args>
            void launch(Args &...args) {
                static_assert(std::is_same_v<std::tuple<Args...>, typename function_trait::Args>);
                void *array[]{(&args)...};
                call(_func, array);
            }
        };
    }
}