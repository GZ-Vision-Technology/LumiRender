//
// Created by Zero on 26/09/2021.
//


#pragma once

#include "render/lights/light.h"

namespace luminous {
    inline namespace render {

        template<typename...T>
        struct BaseBinder2 : public T ... {
            using Bases = std::tuple<T...>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            template<int idx>
            using Base = std::tuple_element_t<idx, Bases>;

            BaseBinder2() = default;

            explicit BaseBinder2(T &&...args)
                    : T{std::move(args)}... {}
        };

        template<>
        struct BaseBinder2<> {
            using Bases = std::tuple<>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            template<int idx>
            using Base = void;

            BaseBinder2() = default;
        };

        template<typename T>
        struct BaseBinder2<T> : public T {
            using Bases = std::tuple<T>;

            static constexpr auto base_num = std::tuple_size_v<Bases>;

            template<int idx>
            using Base = T;

            BaseBinder2() = default;

            using T::T;

            explicit BaseBinder2(T &&t)
                    : T{std::move(t)} {}
        };


        struct LB : BaseBinder2<> {
        public:
//            REFL_CLASS(LB)
        public:
            const LightType _type{};
        public:
            LB()
                    : _type(LightType::Area) {}
        };

        template<typename T>
        struct ICreator2 {
        public:
            CPU_ONLY(
                    template<typename ...Args>
                    static T create(Args &&...args) {
                        return T(std::forward<Args>(args)...);
                    }
            )

            CPU_ONLY(
                    template<typename ...Args>
                    static T *create_ptr(Args &&...args) {
                        return new T(std::forward<Args>(args)...);
                    }
            )
        };

        struct AL : public LB, public luminous::ICreator2<AL> {
        public:
            float padded{};

            AL() : LB() {}
        };
    }
}