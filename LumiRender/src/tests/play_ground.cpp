#include <iostream>

#define DEFINE_FUNC_AND_PTR(ClassName, RetType, FuncName, ...) \
RetType (ClassName::*_fp_##FuncName) __VA_ARGS__ {nullptr};    \
RetType FuncName##_impl __VA_ARGS__ { return ; }

class Base {
public:
    int (Base::*_fp_op)(int a, int b) const{(int(Base::*)(int a, int b)const)&Base::op_impl};
    int op_impl(int a, int b) const {
        return 0;
    }
    template<typename ...Args>
    int op(Args ...args) {
        return (this->*(_fp_op))(std::forward<Args>(args)...);
    }

    DEFINE_FUNC_AND_PTR(Base, void , func, (int a) const)
};


class AddOp: public Base {
public:
    AddOp(int c) {
        this->c = c;
        _fp_op = (int(Base::*)(int, int)const)(&AddOp::op_impl);
    }

protected:
    int op_impl(int a, int b) const {
        return a + b + c;
    }

private:
    int c;
};

int main() {

    Base *p = new AddOp(100);

    std::cout << "AddOp::op result: " << p->op(1,9) << std::endl;
}
