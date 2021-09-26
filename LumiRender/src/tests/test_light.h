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
//        public:
//            int _type{};
//            LB()
//            :BaseBinder3(), _type(0) {}
        };

        struct ICreator2 {
        public:
            int a;
        };

        struct S3 {

        };

        struct AL : public LB,ICreator2, S3 {
        public:
            float padded{9};

            AL() : LB() {
            }
        };
    }
}