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

int foo(int x, int y) {
    std::cout << x << " " << y << std::endl;
    return x;
}

template<typename T>
class TKernel {
public:
    using func_type = T;
    using function_trait = FunctionTrait<func_type>;

    using Functor = decltype(std::function(std::declval<func_type>()));

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

    template<typename Ret, typename...Args, size_t...Is>
    void check_signature(Ret(*f)(Args...), std::index_sequence<Is...>) {
        using OutArgs = std::tuple<Args...>;
        static_assert(std::is_invocable_v<func_type, std::tuple_element_t<Is, OutArgs>...>);
    }

    template<typename...Args>
    void launch_func_impl(Args &&...args) {
        check_signature(_func, std::make_index_sequence<sizeof...(Args)>());
        void *array[]{(&args)...};
        call(_func, array);
    }

    template<typename Ret, typename...Args, typename ...OutArgs>
    void launch_func(Ret(*f)(Args...), OutArgs &&...args) {
        launch_func_impl((static_cast<Args>(std::forward<OutArgs>(args)))...);
    }

    template<typename...Args>
    void launch(Args &&...args) {
        launch_func(_func, std::forward<Args>(args)...);
    }
};


using namespace std;

struct A {

};

int main() {
    int x = 4;
    const float y = 6.5f;

    TKernel kernel(foo);

    cout << typeid(foo).name() << endl;

    A a;

    kernel.launch(x, y);

//    cout << bit_cast<float>(4) << endl;



//    cout << a  << endl;


}