//
// Created by Zero on 26/09/2021.
//


#pragma once

namespace luminous {
    inline namespace render {

        struct BaseBinder3 {

            BaseBinder3() = default;

        };

        struct LB : BaseBinder3 {
        public:
            int _type{};
            LB()
                    : _type(0) {}
        };

        struct ICreator2 {
        public:
        };

        struct AL : public LB, public luminous::ICreator2 {
        public:
            float padded{9};
            float paohui{};

            AL() : LB() {
                padded = 10;
            }
        };
    }
}