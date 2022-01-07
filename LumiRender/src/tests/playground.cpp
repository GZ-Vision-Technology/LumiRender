//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "base_libs/math/common.h"

using namespace std;
using namespace luminous;

int main() {

    auto uc = make_uchar2(155,350);

    cout << float2{uc}.to_string();

    return 0;
}