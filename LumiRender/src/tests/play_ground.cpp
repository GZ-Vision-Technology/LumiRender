//
// Created by Zero on 15/10/2021.
//

#include <iostream>
#include <tuple>
#include <utility>

template<typename Ret, typename...Args, size_t...Is>
void call_impl(Ret(*f)(Args...), void *args[], std::index_sequence<Is...>)
{
    f(*static_cast<std::tuple_element_t<Is, std::tuple<Args...>>*>(args[Is])...);
}

template<typename Ret, typename...Args>
void call(Ret(*f)(Args...), void *args[])
{
    call_impl(f, args, std::make_index_sequence<sizeof...(Args)>());
}

void foo(int x, float y)
{
    std::cout << x << " " << y << std::endl;
}

int main()
{
    int x = 4; float y = 6.5f;
    void *args[] = { &x, &y };
    call(foo, args);
}