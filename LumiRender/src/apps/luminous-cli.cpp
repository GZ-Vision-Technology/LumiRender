//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "core/scene_parser.h"


using namespace std;

int main(int argc, char *argv[]) {
    luminous::Context context{argc, argv};
    if (context.has_help_cmd() || !context.has_scene()) {
        context.print_help();
        return 0;
    }
    try {
        luminous::SceneParser sp;
        sp.load_from_json(context.scene_file());
    } catch (exception e1) {
        cout << e1.what();
        context.print_help();
    }
    return 0;
}