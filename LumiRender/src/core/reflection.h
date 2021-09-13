//
// Created by Zero on 13/09/2021.
//


#pragma once

#include "base_libs/header.h"
#include <map>
#include <string>

namespace luminous {

    class Object {
    public:
        NDSC virtual const char *class_name() const { return type_name(this); }
    };

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

        NDSC size_t size_of(const std::string &class_name) const;

        NDSC size_t size_of(const Object *object) const;

        NDSC size_t align_of(const std::string &class_name) const;

        NDSC size_t align_of(const Object *object) const;

        NDSC size_t type_num() const { return _type_data.size(); }

        void register_class(const std::string &class_name, const TypeData &type_data);

        template<typename T>
        void register_class() {
            auto class_name = type_name<T>();
            TypeData type_data(sizeof(T), alignof(T));
            register_class(class_name, type_data);
        }
    };

    template<typename T>
    class RegisterAction {
    public:
        RegisterAction() {
            ClassFactory::instance()->template register_class<T>();
        }
    };

#define REGISTER(T) RegisterAction<T> Register##T;



}