//
// Created by Zero on 11/12/2021.
//

#include "render/scattering/bxdfs.h"
#include "iostream"

using namespace luminous;
using namespace std;
int main() {

    BxDFs<int, int, std::string> bx_d_fs(1,2, "adfad");

    bx_d_fs.for_each([=](auto obj, int index){
        cout << obj << endl;

        return false;
    });

    return 0;
}