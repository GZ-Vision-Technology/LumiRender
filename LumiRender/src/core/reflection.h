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
        NDSC virtual const char *class_name() const {
            return type_name(this);
        }

        NDSC virtual size_t self_size() const {
            return ClassFactory::instance()->size_of(this);
        }

        NDSC virtual size_t ptr_member_size() const { return 0; }

        NDSC virtual size_t real_size() const {
            return self_size() + ptr_member_size();
        }
    };

#define CALL_SIZE_FUNC(ptr) (+ ((ptr) ? (ptr)->real_size() : 0))
#define GEN_PTR_MEMBER_SIZE_FUNC(...) NDSC size_t ptr_member_size() const override { return MAP(CALL_SIZE_FUNC,__VA_ARGS__) + Super::ptr_member_size(); }


    template<typename T>
    class RegisterAction {
    public:
        RegisterAction() {
            ClassFactory::instance()->template register_class<T>();
        }
    };

#define REGISTER(T) RegisterAction<T> Register##T;


}