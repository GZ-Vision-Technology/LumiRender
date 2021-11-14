#include <exception>
#include <iostream>
#include <type_traits>

template<typename... Ts>
class tagged_pointer {

public:
    template<size_t I, typename Head, typename... Ts>
    struct pointer_element {
        typedef typename pointer_element<I - 1, Ts...>::type type;
    };

    template<typename Header, typename... Ts>
    struct pointer_element<0, Header, Ts...> {
        typedef Header type;
    };

    template<size_t I, typename Head, typename... Ts>
    using pointer_element_t = typename pointer_element<I, Head, Ts...>::type;

    template<typename Tb, typename Td>
    struct dcast_forward {
        typedef std::remove_const_t<Td> dtype;

        static constexpr dtype &&cast(std::remove_reference_t<Tb> &p) {
            return std::forward<dtype>(const_cast<dtype &>(static_cast<const dtype &>(p)));
        }

        static constexpr dtype &&cast(std::remove_reference_t<Tb> &&p) {
            static_assert(!std::is_lvalue_reference_v<dtype>, "bad forward call");
            return std::forward<dtype>(static_cast<dtype &&>(p));
        }
    };

    template<typename Tb, typename Td>
    struct dcast_forward<Tb *, Td> {
        typedef std::remove_const_t<Td> *dtype;

        static constexpr dtype &&cast(std::remove_reference_t<Tb *> &p) {
            return std::forward<dtype>(const_cast<dtype &>(static_cast<const dtype &>(p)));
        }

        static constexpr dtype &&cast(std::remove_reference_t<Tb *> &&p) {
            static_assert(!std::is_lvalue_reference_v<dtype>, "bad forward call");
            return std::forward<dtype>(static_cast<dtype &&>(p));
        }
    };

    template<typename Tb, typename Td>
    struct dcast_forward<const Tb *, Td> {
        typedef Td *dtype;

        static constexpr dtype cast(const Tb *&p) {
            return static_cast<dtype>(const_cast<Tb *>(p));
        }
    };

    template<typename Tb>
    struct tag_getter {
        static int get(const Tb &po) {
            return po.tag();
        }
    };

    template<typename Tb>
    struct tag_getter<Tb *> {
        static int get(const Tb *po) {
            return po->tag();
        }
    };

    template<size_t N>
    struct dispatch_seq {

        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            std::invoke_result_t<Fn, Ty, Targs...> result;
            int check_ret = ((tag_getter<Ty>::get(po) == Ts::type_tag()
                              ? (result = fn(dcast_forward<Ty, Ts>::cast(po), std::forward<Targs>(args)...), 1)
                              : 0) ||
                    ...);
            if (!check_ret)
                throw std::invalid_argument("unknown type");

            return result;
        }
    };

#define _DELEGATING_CASE(I)                                                                                            \
case pointer_element_t<I, Ts...>::type_tag(): {                                                                      \
using Tyd = pointer_element_t<I, Ts...>;                                                                           \
return fn(dcast_forward<Ty, Tyd>::cast(po), std::forward<Targs>(args)...);                                         \
}

    template<>
    struct dispatch_seq<1> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (po->tag()) { _DELEGATING_CASE(0)}
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<2> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<3> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<4> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<5> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<6> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<7> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<8> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
                _DELEGATING_CASE(7)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<9> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
                _DELEGATING_CASE(7)
                _DELEGATING_CASE(8)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<10> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
                _DELEGATING_CASE(7)
                _DELEGATING_CASE(8)
                _DELEGATING_CASE(9)
                _DELEGATING_CASE(10)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<11> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
                _DELEGATING_CASE(7)
                _DELEGATING_CASE(8)
                _DELEGATING_CASE(9)
                _DELEGATING_CASE(10)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<>
    struct dispatch_seq<12> {
        template<typename Ty, typename Fn, typename... Targs>
        static std::invoke_result_t<Fn, Ty, Targs...> invoke(Ty &&po, Fn &&fn, Targs &&...args) {
            switch (tag_getter<Ty>::get(po)) {
                _DELEGATING_CASE(0)
                _DELEGATING_CASE(1)
                _DELEGATING_CASE(2)
                _DELEGATING_CASE(3)
                _DELEGATING_CASE(4)
                _DELEGATING_CASE(5)
                _DELEGATING_CASE(6)
                _DELEGATING_CASE(7)
                _DELEGATING_CASE(8)
                _DELEGATING_CASE(9)
                _DELEGATING_CASE(10)
                _DELEGATING_CASE(11)
            }
            throw std::invalid_argument("unknown type");
        }
    };

    template<typename Ty, typename Fn, typename... Targs>
    static std::invoke_result_t<Fn, Ty, Targs...> dispatch(Ty &&po, Fn &&fn, Targs &&...args) {
        return dispatch_seq<sizeof...(Ts)>::invoke(std::forward<Ty>(po), std::forward<Fn>(fn),
                                                   std::forward<Targs>(args)...);
    }
};

class add_op;

class sub_op;

class mul_op;

class div_op;

class binary_op {

public:
    enum {
        TYPE_TAG = -1
    };

    using dispatcher = tagged_pointer<add_op, sub_op, mul_op, div_op>;

    static constexpr int type_tag = TYPE_TAG;

    int tag() const {
        return tag_;
    }

    float operator()(float a, float b) const {

        return dispatcher::dispatch(
                *this, [](const auto po, float a, float b) { return po(a, b); }, a, b);
    }

    float another(float a, float b) const {
        return dispatcher::dispatch(
                this, [](const auto po, float a, float b) { return (*po)(a, b); }, a, b);
    }

protected:
    int tag_ = TYPE_TAG;
};

class add_op : public binary_op {

    enum {
        TYPE_TAG = 0
    };

public:
    add_op() {
        tag_ = TYPE_TAG;
    }

    static constexpr int type_tag() {
        return TYPE_TAG;
    }

    float operator()(float a, float b) const {
        std::cout << "Call add_op::operator()" << std::endl;

        return a + b;
    }
};

class sub_op : public binary_op {
    enum {
        TYPE_TAG = 1
    };

public:
    sub_op() {
        tag_ = TYPE_TAG;
    }

    static constexpr int type_tag() {
        return TYPE_TAG;
    };

    float operator()(float a, float b) const {
        std::cout << "Call sub_op::operator()" << std::endl;

        return a - b;
    }
};

class mul_op : public binary_op {
    enum {
        TYPE_TAG = 2
    };

public:
    mul_op() {
        tag_ = TYPE_TAG;
    }

    static constexpr int type_tag() {
        return TYPE_TAG;
    }

    float operator()(float a, float b) const {
        std::cout << "Call mul_op::operator()" << std::endl;

        return a * b;
    }
};

class div_op : public binary_op {
    enum {
        TYPE_TAG = 3
    };

public:
    div_op() {
        tag_ = TYPE_TAG;
    }

    static constexpr int type_tag() {
        return TYPE_TAG;
    }

    float operator()(float a, float b) const {
        std::cout << "Call div_op::operator()" << std::endl;

        return a / b;
    }
};

int main() {

    binary_op *test_ops[4] = {new add_op(), new sub_op(), new mul_op(), new div_op()};

    for (auto op : test_ops) {
        std::cout << "Result: " << op->operator()(5.0, 3.0) << std::endl;

        std::cout << "Another approach: " << op->another(5.0, -3.0) << std::endl;
        delete op;
    }
}
