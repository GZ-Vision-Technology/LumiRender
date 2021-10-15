//
// Created by Zero on 15/10/2021.
//

#include <iostream>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>

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

    template<int idx>
    using arg_type = std::tuple_element_t<idx, Args>;
};

template<typename func_type>
class Kernel {
public:
    using function_trait = FunctionTrait<func_type>;
protected:
    func_type _func{};
public:
    explicit Kernel(func_type func) : _func(func) {}

    template<typename Ret, typename...Args, size_t...Is>
    void call_impl(Ret(*f)(Args...), void *args[], std::index_sequence<Is...>) {
        f(*static_cast<std::tuple_element_t<Is, std::tuple<Args...>> *>(args[Is])...);
    }

    template<typename Ret, typename...Args, size_t...Is>
    void check_signature(Ret(*f)(Args...), std::index_sequence<Is...>) {
//        static_assert();
    }

    template<typename Ret, typename...Args>
    void call(Ret(*f)(Args...), void *args[]) {
        call_impl(f, args, std::make_index_sequence<sizeof...(Args)>());
    }

    template<typename...Args>
    void launch(Args &...args) {
        using ArgsTuple = std::tuple<Args...>;

        static_assert(std::is_same_v<ArgsTuple, function_trait::Args>);

        void *array[]{(&args)...};
        call(_func, array);
    }
};

using namespace std;


int main() {
    int x = 4;
    float y = 6.5f;

    Kernel kernel(foo);

    auto func = std::function(foo);

    func(x, y);

    cout << typeid(FunctionTrait<decltype(foo)>::Args).name() << endl;

    kernel.launch(x, y);
}