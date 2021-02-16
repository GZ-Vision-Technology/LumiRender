//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"


using namespace std;

int main(int argc, char *argv[]) {
    luminous::Context context{argc, argv};
    context.print_help();
    cout << context.scene_path();
//    cout << context.input_path() << endl;
//    cout << context.runtime_path() << endl;
//    cout << context.cache_path() << endl;
//    cout << context.working_path() << endl;
    return 0;
}