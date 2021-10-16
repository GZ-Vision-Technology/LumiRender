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

template<typename T>
class CUDAKernel : TKernel<T> {

public:
    template<typename...Args>
    void launch(Args &...args) {
        printf("adfafasfsd");
        static_assert(std::is_same_v<std::tuple<Args...>, typename function_trait::Args>);
        void *array[]{(&args)...};
        call(_func, array);
    }

};


using namespace std;


int main() {
    int x = 4;
    float y = 6.5f;

    TKernel kernel(foo);

    kernel.launch(x, y);


}