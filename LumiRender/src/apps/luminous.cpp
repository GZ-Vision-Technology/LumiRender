//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "util/parser.h"
#include <memory>
#include "view/application.h"

using std::cout;
using std::endl;
using namespace luminous;

int execute(int argc, char *argv[]) {
    logging::set_log_level(spdlog::level::info);
    Context context{argc, argv};
    context.try_print_help_and_exit();
    if (argc == 1) {
        context.print_help();
        return 0;
    }
    Parser sp(&context);
    try {
        if (context.has_scene()) {
            sp.load(context.scene_file());
        }
    } catch (std::exception &e1) {
        cout << e1.what();
    }

    App app("luminous", luminous::make_int2(1280,720), &context, sp);
    return app.run();
}

int main(int argc, char *argv[]) {
    auto ret = execute(argc, argv);
    return ret;
}