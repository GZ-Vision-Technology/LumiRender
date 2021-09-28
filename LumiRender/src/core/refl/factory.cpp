//
// Created by Zero on 13/09/2021.
//

#include "factory.h"
#include "core/logging.h"

namespace luminous {

    ClassFactory *ClassFactory::_instance = nullptr;

    ClassFactory *ClassFactory::instance() {
        if (_instance == nullptr) {
            _instance = new ClassFactory();
        }
        return _instance;
    }

    size_t ClassFactory::runtime_size_of(const Object *object) const {
        return object == nullptr ? 0 : size_of(object);
    }

    void ClassFactory::register_class(const std::string &class_name, const TypeData &type_data) {
        _type_data.insert(std::make_pair(class_name, type_data));
    }

    bool ClassFactory::is_registered(const std::string &class_name) const {
        auto iter = _type_data.find(class_name);
        return iter != _type_data.cend();
    }

    bool ClassFactory::is_registered(const Object *object) const {
        return is_registered(type_name(object));
    }

    const TypeData& ClassFactory::type_data(const Object *object) const {
        return type_data(type_name(object));
    }

    const TypeData& ClassFactory::type_data(const std::string &class_name) const {
        DCHECK(is_registered(class_name));
        return _type_data.at(class_name);
    }
    
}