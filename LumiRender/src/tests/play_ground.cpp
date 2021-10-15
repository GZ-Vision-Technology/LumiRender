//
// Created by Zero on 15/10/2021.
//

#include <iostream>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>


template<typename func_type>
class Kernel {
protected:
    func_type _func{};
public:
    explicit Kernel(func_type func) : _func(func) {}

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
        void *array[]{(&args)...};
        call(_func, array);
    }
};

int foo(int x, float y) {
    std::cout << x << " " << y << std::endl;
    return x;
}

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

using namespace std;


int main() {
    int x = 4;
    float y = 6.5f;
    void *args[] = {&x, &y};

    Kernel kernel(foo);

    auto func = std::function(foo);

    func(x, y);

    cout << typeid(FunctionTrait<decltype(foo)>::Args).name() << endl;

    kernel.launch(x, y);
}