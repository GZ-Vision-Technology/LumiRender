//
// Created by Zero on 15/10/2021.
//

#include <iostream>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>
#include "base_libs/common.h"
#include "base_libs/lstd/lstd.h"


using namespace luminous;

int foo(int x, float y) {
    std::cout << x << " " << y << std::endl;
    return x;
}

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
    void check_signature(Ret(*f)(Args...), std::index_sequence<Is...>) {
        using OutArgs = std::tuple<Args...>;
        static_assert(std::is_invocable_v<func_type, std::tuple_element_t<Is, OutArgs>...>);
    }

    template<typename Ret, typename...Args, size_t...Is>
    void call_impl(Ret(*f)(Args...), void *args[], std::index_sequence<Is...>) {
        f(*static_cast<std::tuple_element_t<Is, std::tuple<Args...>> *>(args[Is])...);
    }

    template<typename Ret, typename...Args>
    void call(Ret(*f)(Args...), void *args[]) {
        call_impl(f, args, std::make_index_sequence<sizeof...(Args)>());
    }

    template<typename...Args>
    void launch(Args &&...args) {
        check_signature(_func, std::make_index_sequence<sizeof...(Args)>());
        void *array[]{(&args)...};
        call(_func, array);
    }
};


using namespace std;


int main() {
    int x = 4;
    float y = 6.5f;

    TKernel kernel(foo);

    cout << typeid(foo).name() << endl;

    kernel.launch(x, x);

    cout << bit_cast<float>(4) << endl;


}