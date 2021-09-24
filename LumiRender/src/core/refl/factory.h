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

            LM_NODISCARD const TypeData &type_data(const Object *object) const;

            LM_NODISCARD const TypeData &type_data(const std::string &class_name) const;

            template<typename ...Args>
            LM_NODISCARD const std::string &super_class_name(Args &&...args) const {
                return type_data(std::forward<Args>(args)...).super_class;
            }

            template<typename ...Args>
            LM_NODISCARD const std::vector<uint32_t> &member_offsets(Args &&...args) const {
                return type_data(std::forward<Args>(args)...).member_offsets;
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

#define GET_VALUE(ptr, offset) (reinterpret_cast<uint64_t*>(&((reinterpret_cast<std::byte *>(ptr))[offset]))[0])

#define SET_VALUE(ptr, offset, val) reinterpret_cast<uint64_t*>(&((reinterpret_cast<std::byte*>(ptr))[offset]))[0] = val

        class Object : public BaseBinder<>{
        public:
            REFL_CLASS(Object)

            uint64_t get_value(uint32_t offset) {
                return GET_VALUE(this, offset);
            }

            void set_value(uint32_t offset, uint64_t val) {
                SET_VALUE(this, offset, val);
            }
        };

        template<typename T>
        class RegisterAction {
        public:
            RegisterAction() {
                TypeData &type_data = ClassFactory::instance()->template register_class<T>();
                for_each_all_registered_member<T>([&](auto offset, auto name, auto ptr) {
                    type_data.member_offsets.push_back(offset);
                });
            }
        };

#define REGISTER(T) RegisterAction<T> Register##T;

#define DEFINE_CLASS_BEGIN(ClassName, mode, Super, ...) \
    class ClassName : mode Super,__VA_ARGS__ {          \
    public:                                             \
        REFL_CLASS(ClassName)

#define DEFINE_CLASS_END(ClassName) }; REGISTER(ClassName);

    }

}