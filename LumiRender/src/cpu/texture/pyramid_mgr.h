//
// Created by Zero on 07/01/2022.
//


#pragma once

#include "util/image_base.h"
#include "core/memory/block_array.h"

namespace luminous {
    inline namespace cpu {

        template<typename T>
        class Pyramid {
        public:
            using element_ty = std::unique_ptr<BlockedArray<T>>;
            using vector_ty = std::vector<element_ty>;
        private:
            vector_ty _vector;
        public:
            Pyramid() = default;

            LM_NODISCARD size_t size_in_bytes() const {
                size_t ret = 0u;
                for (const auto &elm : _vector) {
                    ret += elm->size_in_bytes();
                }
                return ret;
            }

            void add_layer(const element_ty &element) {
                _vector.push_back(element);
            }

            LM_NODISCARD size_t levels() const { return _vector.size(); }

            void clear() { _vector.clear(); }

            LM_NODISCARD static element_ty construct_layer(uint2 res, std::byte *data) {
//                T * ptr = create<>()
            }
        };

        class PyramidMgr {
        private:
            template<typename T>
            using PyramidVector = std::vector<Pyramid<T>>;

            static PyramidMgr *_instance;

            PyramidVector<uchar> _pyramid_vector_uc;
            PyramidVector<uchar2> _pyramid_vector_uc2;
            PyramidVector<uchar4> _pyramid_vector_uc4;

            PyramidVector<float> _pyramid_vector_float;
            PyramidVector<float2> _pyramid_vector_float2;
            PyramidVector<float4> _pyramid_vector_float4;

            template<typename T>
            LM_NODISCARD PyramidVector<T> &get_vector() {
                if constexpr (std::is_same_v<T, char>) {
                    return _pyramid_vector_uc;
                } else if constexpr(std::is_same_v<T, char2>) {
                    return _pyramid_vector_uc2;
                } else if constexpr(std::is_same_v<T, char4>) {
                    return _pyramid_vector_uc4;
                } else if constexpr(std::is_same_v<T, float>) {
                    return _pyramid_vector_float;
                } else if constexpr(std::is_same_v<T, float2>) {
                    return _pyramid_vector_float2;
                } else if constexpr(std::is_same_v<T, float4>) {
                    return _pyramid_vector_float4;
                }
            }

            template<typename T>
            LM_NODISCARD size_t _size_in_bytes(const PyramidVector<T> &vector) const {
                size_t ret = 0u;
                for (const Pyramid<T> &elm : vector) {
                    ret += elm.size_in_bytes();
                }
                return ret;
            }

            template<typename T>
            LM_NODISCARD const PyramidVector<T> &get_vector() const {
                if constexpr (std::is_same_v<T, char>) {
                    return _pyramid_vector_uc;
                } else if constexpr(std::is_same_v<T, char2>) {
                    return _pyramid_vector_uc2;
                } else if constexpr(std::is_same_v<T, char4>) {
                    return _pyramid_vector_uc4;
                } else if constexpr(std::is_same_v<T, float>) {
                    return _pyramid_vector_float;
                } else if constexpr(std::is_same_v<T, float2>) {
                    return _pyramid_vector_float2;
                } else if constexpr(std::is_same_v<T, float4>) {
                    return _pyramid_vector_float4;
                }
            }

            PyramidMgr() = default;

            PyramidMgr(const PyramidMgr &other) = default;

            PyramidMgr(PyramidMgr &&other) = default;

        public:
            static PyramidMgr *instance();

            void clear() {
                _pyramid_vector_uc.clear();
                _pyramid_vector_uc2.clear();
                _pyramid_vector_uc4.clear();
                _pyramid_vector_float.clear();
                _pyramid_vector_float2.clear();
                _pyramid_vector_float4.clear();
            }

            LM_NODISCARD size_t size_in_bytes() const {
                size_t ret{0u};
                ret += _size_in_bytes(_pyramid_vector_uc);
                ret += _size_in_bytes(_pyramid_vector_uc2);
                ret += _size_in_bytes(_pyramid_vector_uc4);
                ret += _size_in_bytes(_pyramid_vector_float);
                ret += _size_in_bytes(_pyramid_vector_float2);
                ret += _size_in_bytes(_pyramid_vector_float4);
                return ret;
            }

            template<typename T>
            LM_NODISCARD index_t add_pyramid(Pyramid<T> &&pyramid) {
                PyramidVector<T> &Vector = get_vector<T>();
                Vector.push_back(pyramid);
                return Vector.size();
            }

            template<typename T>
            LM_NODISCARD const Pyramid<T> &get_pyramid(index_t index) const {
                return get_vector<T>().at(index);
            }
        };
    }
}