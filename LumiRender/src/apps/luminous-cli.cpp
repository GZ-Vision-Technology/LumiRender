//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "core/scene_parser.h"


using namespace std;

int main(int argc, char *argv[]) {
    luminous::Context context{argc, argv};
    if (context.has_help_cmd()) {
        context.print_help();
        return 0;
    }
    if (!context.has_scene()) {
        context.print_help();
        return 0;
    }
    cout << context.scene_file();
    luminous::SceneParser sp;
    sp.load_from_json(context.scene_file());
    return 0;
}