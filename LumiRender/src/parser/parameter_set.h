//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "base_libs/header.h"
#include "base_libs/math/common.h"
#include <vector>

namespace luminous {
    inline namespace utility {
        using std::string;

        class ParameterSet {
        private:
            std::string _key;
            DataWrap _data;
        private:


#define LUMINOUS_MAKE_AS_TYPE_FUNC(type) LM_NODISCARD type _as_##type() const  {   \
            return static_cast<type>(_data);                                                \
        }

#define LUMINOUS_MAKE_AS_TYPE_VEC2(type) LM_NODISCARD type##2 _as_##type##2() const  { \
            return make_##type##2(this->at(0).as_##type(), this->at(1).as_##type());                                    \
        }                                                                                       \

#define LUMINOUS_MAKE_AS_TYPE_VEC3(type) LM_NODISCARD type##3 _as_##type##3() const  { \
            return make_##type##3(this->at(0).as_##type(), this->at(1).as_##type(), this->at(2).as_##type());     \
        }
#define LUMINOUS_MAKE_AS_TYPE_VEC4(type) LM_NODISCARD type##4 _as_##type##4() const  { \
            return make_##type##4(this->at(0).as_##type(), this->at(1).as_##type(), this->at(2).as_##type(), this->at(3).as_##type());          \
        }                                                                                       \
        template<typename T, std::enable_if_t<std::is_same_v<T, type##4>, int> = 0>            \
        T _as() const {                                                                     \
            return _as_##type##4();                                                            \
        }
#define LUMINOUS_MAKE_AS_TYPE_VEC(type)   \
        LUMINOUS_MAKE_AS_TYPE_VEC2(type)  \
        LUMINOUS_MAKE_AS_TYPE_VEC3(type)  \
        LUMINOUS_MAKE_AS_TYPE_VEC4(type)


#define LUMINOUS_MAKE_AS_TYPE_MAT(type)    \
        LUMINOUS_MAKE_AS_TYPE_MAT3X3(type) \
        LUMINOUS_MAKE_AS_TYPE_MAT4X4(type)


#define LUMINOUS_MAKE_AS_TYPE_MAT3X3(type) LM_NODISCARD float3x3 _as_##type##3x3() const  { \
            if (_data.size() == 3) {                                                                 \
                return make_##type##3x3(                                                             \
                        this->at(0)._as_##type##3(),                                                 \
                        this->at(1)._as_##type##3(),                                                 \
                        this->at(2)._as_##type##3());                                                \
            } else {                                                                                 \
               return make_##type##3x3(this->at(0)._as_##type(),                                     \
                        this->at(1)._as_##type(),                                                    \
                        this->at(2)._as_##type(),                                                    \
                        this->at(3)._as_##type(),                                                    \
                        this->at(4)._as_##type(),                                                    \
                        this->at(5)._as_##type(),                                                    \
                        this->at(6)._as_##type(),                                                    \
                        this->at(7)._as_##type(),                                                    \
                        this->at(8)._as_##type());                                                   \
            }                                                                                        \
        }
#define LUMINOUS_MAKE_AS_TYPE_MAT4X4(type) LM_NODISCARD float4x4 _as_##type##4x4() const  { \
            if (_data.size() == 4) {                                                                 \
                return make_##type##4x4(                                                             \
                        this->at(0)._as_##type##4(),                                                 \
                        this->at(1)._as_##type##4(),                                                 \
                        this->at(2)._as_##type##4(),                                                 \
                        this->at(3)._as_##type##4());                                                \
            } else {                                                                                 \
                return make_##type##4x4(this->at(0)._as_##type(),                                    \
                                        this->at(1)._as_##type(),                                    \
                                        this->at(2)._as_##type(),                                    \
                                        this->at(3)._as_##type(),                                    \
                                        this->at(4)._as_##type(),                                    \
                                        this->at(5)._as_##type(),                                    \
                                        this->at(6)._as_##type(),                                    \
                                        this->at(7)._as_##type(),                                    \
                                        this->at(8)._as_##type(),                                    \
                                        this->at(9)._as_##type(),                                    \
                                        this->at(10)._as_##type(),                                   \
                                        this->at(11)._as_##type(),                                   \
                                        this->at(12)._as_##type(),                                   \
                                        this->at(13)._as_##type(),                                   \
                                        this->at(14)._as_##type(),                                   \
                                        this->at(15)._as_##type());                                  \
            }                                                                                        \
        }

            LUMINOUS_MAKE_AS_TYPE_FUNC(int)

            LUMINOUS_MAKE_AS_TYPE_FUNC(uint)

            LUMINOUS_MAKE_AS_TYPE_FUNC(bool)

            LUMINOUS_MAKE_AS_TYPE_FUNC(float)

            LUMINOUS_MAKE_AS_TYPE_FUNC(string)

            LUMINOUS_MAKE_AS_TYPE_VEC(uint)

            LUMINOUS_MAKE_AS_TYPE_VEC(int)

            LUMINOUS_MAKE_AS_TYPE_VEC(float)

            LUMINOUS_MAKE_AS_TYPE_MAT(float)

#undef LUMINOUS_MAKE_AS_TYPE_FUNC

#undef LUMINOUS_MAKE_AS_TYPE_VEC
#undef LUMINOUS_MAKE_AS_TYPE_VEC2
#undef LUMINOUS_MAKE_AS_TYPE_VEC3
#undef LUMINOUS_MAKE_AS_TYPE_VEC4

#undef LUMINOUS_MAKE_AS_TYPE_MAT
#undef LUMINOUS_MAKE_AS_TYPE_MAT3X3
#undef LUMINOUS_MAKE_AS_TYPE_MAT4X4


        public:
            ParameterSet() = default;

            explicit ParameterSet(const DataWrap &json,
                                  const string &key = "") :
                    _data(json),
                    _key(key) {}

            void setJson(const DataWrap &json) { _data = json; }

            LM_NODISCARD DataWrap data() const { return _data; }

            LM_NODISCARD ParameterSet get(const std::string &key) const {
                return ParameterSet(_data[key], key);
            }

            LM_NODISCARD ParameterSet at(uint idx) const {
                return ParameterSet(_data.at(idx));
            }

            LM_NODISCARD ParameterSet operator[](const std::string &key) const {
                return ParameterSet(_data.value(key, DataWrap()), key);
            }

            LM_NODISCARD bool contains(const std::string &key) const {
                return _data.contains(key);
            }

            LM_NODISCARD ParameterSet operator[](uint i) const {
                return ParameterSet(_data[i]);
            }

            template<typename T>
            LM_NODISCARD std::vector<T> as_vector() const {
                LUMINOUS_EXCEPTION_IF(!_data.is_array(), "data is not array!");
                std::vector<T> ret;
                for (const auto &elm : _data) {
                    ParameterSet ps{elm};
                    ret.push_back(ps.template as<T>());
                }
                return ret;
            }

#define LUMINOUS_MAKE_AS_TYPE_SCALAR(type) LM_NODISCARD type as_##type(type val = type()) const {                   \
            try {                                                                                               \
                return _as_##type();                                                                            \
            } catch (const std::exception &e) {                                                             \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type,",key is ",_key, ", using default value: \"", val, "\""); \
                return val;                                                                                     \
            }                                                                                                   \
        }                                                                                                       \
        template<typename T, std::enable_if_t<std::is_same_v<T, type>, int> = 0>           \
        T as() const {                                                                     \
            return as_##type();                                                            \
        }\

#define LUMINOUS_MAKE_AS_TYPE_VEC2(type) LM_NODISCARD type##2 as_##type##2(type##2 val = make_##type##2()) const noexcept {      \
            try {                                                                                                        \
                return _as_##type##2();                                                                                  \
            } catch (const std::exception &e) {                                                                      \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type,",key is ",_key, ", using default value: \"(", val.to_string() , ")\""); \
                return val;                                                                                              \
            }                                                                                                            \
        } \
        template<typename T, std::enable_if_t<std::is_same_v<T, type##2>, int> = 0>            \
        T as() const {                                                                     \
            return as_##type##2();                                                            \
        }
#define LUMINOUS_MAKE_AS_TYPE_VEC3(type) LM_NODISCARD type##3 as_##type##3(type##3 val = make_##type##3()) const noexcept {        \
            try {                                                                                                          \
                return _as_##type##3();                                                                                    \
            } catch (const std::exception &e) {                                                                        \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type,",key is ",_key, ", using default value: \"(", val.to_string() , ")\""); \
                return val;                                                                                              \
            }                                                                                                            \
        } \
        template<typename T, std::enable_if_t<std::is_same_v<T, type##3>, int> = 0>            \
        T as() const {                                                                     \
            return as_##type##3();                                                            \
        }
#define LUMINOUS_MAKE_AS_TYPE_VEC4(type) LM_NODISCARD type##4 as_##type##4(type##4 val = make_##type##4()) const noexcept{        \
            try {                                                                                                          \
                return _as_##type##4();                                                                                    \
            } catch (const std::exception &e) {                                                                        \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type,",key is ",_key, ", using default value: \"(", val.to_string() , ")\""); \
                return val;                                                                                              \
            }                                                                                                            \
        } \
        template<typename T, std::enable_if_t<std::is_same_v<T, type##4>, int> = 0>            \
        T as() const {                                                                     \
            return as_##type##4();                                                            \
        }
#define LUMINOUS_MAKE_AS_TYPE_MAT3X3(type) LM_NODISCARD type##3x3 as_##type##3x3(type##3x3 val = make_##type##3x3()) const noexcept{ \
            try {                                                                                                                       \
                return _as_##type##3x3(); \
            } catch (const std::exception &e) { \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type,",key is ",_key, ", using default value: \"(", val.to_string() , ")\""); \
                return val; \
            } \
        }
#define LUMINOUS_MAKE_AS_TYPE_MAT4X4(type) LM_NODISCARD type##4x4 as_##type##4x4(type##4x4 val = make_##type##4x4()) const noexcept { \
            try {                                                                                                                       \
                return _as_##type##4x4(); \
            } catch (const std::exception &e) { \
                LUMINOUS_WARNING("Error occurred while parsing parameter type is ", #type, ",key is ",_key,", using default value: \"(", val.to_string() , ")\""); \
                return val; \
            } \
        }

            LUMINOUS_MAKE_AS_TYPE_MAT3X3(float)

            LUMINOUS_MAKE_AS_TYPE_MAT4X4(float)

            LUMINOUS_MAKE_AS_TYPE_SCALAR(float)

            LUMINOUS_MAKE_AS_TYPE_SCALAR(uint)

            LUMINOUS_MAKE_AS_TYPE_SCALAR(bool)

            LUMINOUS_MAKE_AS_TYPE_SCALAR(int)

            LUMINOUS_MAKE_AS_TYPE_SCALAR(string)

#define LUMINOUS_MAKE_AS_TYPE_VEC(type)  \
        LUMINOUS_MAKE_AS_TYPE_VEC2(type) \
        LUMINOUS_MAKE_AS_TYPE_VEC3(type) \
        LUMINOUS_MAKE_AS_TYPE_VEC4(type)

            LUMINOUS_MAKE_AS_TYPE_VEC(int)

            LUMINOUS_MAKE_AS_TYPE_VEC(uint)

            LUMINOUS_MAKE_AS_TYPE_VEC(float)


#undef LUMINOUS_MAKE_AS_TYPE_SCALAR
#undef LUMINOUS_MAKE_AS_TYPE_VEC
#undef LUMINOUS_MAKE_AS_TYPE_VEC2
#undef LUMINOUS_MAKE_AS_TYPE_VEC3
#undef LUMINOUS_MAKE_AS_TYPE_VEC4
#undef LUMINOUS_MAKE_AS_TYPE_MAT3X3
#undef LUMINOUS_MAKE_AS_TYPE_MAT4X4
        };
    } //luminous::render
} // luminous