//
// Created by Zero on 13/09/2021.
//


#pragma once

#include "base_libs/header.h"
#include <map>
#include <string>
#include "core/macro_map.h"

namespace luminous {

    class Object;

    struct TypeData {

        const size_t size;
        const size_t align;

        explicit TypeData(size_t size = 0, size_t align = 0)
                : size(size), align(align) {}
    };

    class ClassFactory {
    private:
        static ClassFactory *_instance;

        std::map<std::string, TypeData> _type_data;

        ClassFactory() = default;

        ClassFactory(const ClassFactory &) = default;

        ClassFactory &operator=(const ClassFactory &) = default;

    public:
        static ClassFactory *instance();

        NDSC bool is_registered(const std::string &class_name) const;

        NDSC bool is_registered(const Object *object) const;

        NDSC size_t size_of(const std::string &class_name) const;

        NDSC size_t runtime_size_of(const Object *object) const;

        NDSC size_t size_of(const Object *object) const;

        NDSC size_t align_of(const std::string &class_name) const;

        NDSC size_t align_of(const Object *object) const;

        NDSC TypeData type_data(const Object *object) const;

        NDSC TypeData type_data(const std::string &class_name) const;

        NDSC size_t type_num() const { return _type_data.size(); }

        void register_class(const std::string &class_name, const TypeData &type_data);

        template<typename T>
        void register_class() {
            auto class_name = type_name<T>();
            TypeData type_data(sizeof(T), alignof(T));
            register_class(class_name, type_data);
        }

        template<typename ...Args>
        size_t size_sum(Args &&...args) {

        }
    };

    class Object {
    public:
        using PtrMapper = std::map<Object *, size_t>;

        NDSC virtual const char *class_name() const {
            return type_name(this);
        }

        NDSC virtual size_t self_size() const {
            return ClassFactory::instance()->size_of(this);
        }

        NDSC virtual size_t ptr_member_size() const { return 0; }

        template<typename ...Args>
        NDSC static size_t size_sum(Args &&...ptr) {
            return (... + (ptr == nullptr ? 0 : ptr->real_size()));
        }

        NDSC virtual size_t real_size() const {
            return self_size() + ptr_member_size();
        }

        template<typename ...Args>
        static size_t mapping_ptr(PtrMapper &mapper, size_t offset, Args &&...ptr) {
            ((mapper[ptr] = offset, offset += ptr->mapping_member_ptr()), ...);
            return offset;
        }

        virtual size_t mapping_member_ptr(PtrMapper &mapper, size_t offset) {
            mapper[this] = offset;
            offset += self_size();
            return offset;
        }
    };

#define GEN_PTR_MEMBER_SIZE_FUNC(Super, ...) NDSC size_t ptr_member_size() const override { \
        return size_sum(__VA_ARGS__) + Super::ptr_member_size();                            \
    }

#define GEN_PTR_MEMBER_SIZE

#define DEFINE_PTR_VAR(Statement) Statement{};
#define DEFINE_PTR_VARS(...) MAP(DEFINE_PTR_VAR, __VA_ARGS__)


    template<typename T>
    class RegisterAction {
    public:
        RegisterAction() {
            ClassFactory::instance()->template register_class<T>();
        }
    };

#define REGISTER(T) RegisterAction<T> Register##T;


    class A : public Object {
    public:
        int b;
        A *pa{};
        using Super = Object;

        GEN_PTR_MEMBER_SIZE_FUNC(Object, pa)



        size_t mapping_member_ptr(PtrMapper &mapper, size_t offset) override {
            offset = Object::mapping_member_ptr(mapper, offset);
            offset = mapping_ptr(mapper, offset, pa);
            return offset;
        }
    };

    class B : public A {
    public:
        int b;
    };

    class C : public A {
    public:
        int c;

        DEFINE_PTR_VARS(Object *pc, Object *p1)

        GEN_PTR_MEMBER_SIZE_FUNC(A, pc, p1)



    };

    REGISTER(B)
    REGISTER(A)
    REGISTER(C)


}