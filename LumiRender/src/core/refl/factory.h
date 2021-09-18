//
// Created by Zero on 13/09/2021.
//


#pragma once

#include "base_libs/header.h"
#include <map>
#include <vector>
#include <string>
#include "reflection.h"
#include "core/macro_map.h"
#include "base_libs/math/interval.h"

namespace luminous {

    inline namespace refl {
        class Object;

        struct TypeData {
            size_t size;
            size_t align;
            std::vector<size_t> member_offsets;
            std::string super_class;

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

            LM_NODISCARD bool is_registered(const std::string &class_name) const;

            LM_NODISCARD bool is_registered(const Object *object) const;

            LM_NODISCARD size_t runtime_size_of(const Object *object) const;

            template<typename ...Args>
            LM_NODISCARD size_t size_of(Args &&...args) const {
                return type_data(std::forward<Args>(args)...).size;
            }

            template<typename ...Args>
            LM_NODISCARD size_t align_of(Args &&...args) const {
                return type_data(std::forward<Args>(args)...).align;
            }

            LM_NODISCARD const TypeData& type_data(const Object *object) const;

            LM_NODISCARD const TypeData& type_data(const std::string &class_name) const;

            template<typename ...Args>
            LM_NODISCARD const std::string &super_class_name(Args &&...args) {
                return type_data(std::forward<Args>(args)...).super_class;
            }

            LM_NODISCARD size_t type_num() const { return _type_data.size(); }

            void register_class(const std::string &class_name, const TypeData &type_data);

            template<typename T>
            TypeData &register_class() {
                auto class_name = type_name<T>();
                TypeData td(sizeof(T), alignof(T));
                register_class(class_name, td);
                return _type_data[class_name];
            }
        };

        #define DECLARE_SUPER(ClassName) using Super = ClassName;

        class Empty {};

        class Object : public Empty {
        public:
            REFL_CLASS(Object)

            DECLARE_SUPER(Empty)

            LM_NODISCARD virtual const char *class_name() const {
                return type_name(this);
            }

            LM_NODISCARD virtual std::string super_class_name() const {
                return ClassFactory::instance()->super_class_name(this);
            }

            LM_NODISCARD virtual size_t self_size() const {
                return ClassFactory::instance()->size_of(this);
            }

            LM_NODISCARD virtual size_t ptr_member_size() const {
                size_t ret = 0;
                TypeData type_data = ClassFactory::instance()->type_data(this);



                return ret;
            }

            LM_NODISCARD virtual size_t real_size() const {
                return self_size() + ptr_member_size();
            }
        };

        template<typename T>
        class RegisterAction {
        public:
            RegisterAction() {
                TypeData &type_data = ClassFactory::instance()->template register_class<T>();
                using Super = typename T::Super;
                type_data.super_class = type_name<Super>();
                for_each_ptr_member<T>([&](auto offset, auto name) {
                    type_data.member_offsets.push_back(offset);
                });
            }
        };

#define REGISTER(T) RegisterAction<T> Register##T;

#define DEFINE_CLASS_BEGIN(ClassName, mode, Super, ...) \
    class ClassName : mode Super,__VA_ARGS__ {          \
    public:                                             \
        DECLARE_SUPER(Super)                            \
        REFL_CLASS(ClassName)

#define DEFINE_CLASS_END(ClassName) }; REGISTER(ClassName);

    }

}