//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "render/include/parser.h"
#include <memory>
#include "view/application.h"
#include <glad/glad.h>
#include<chrono>
#include "gpu/framework/cuda_pipeline.h"

using namespace std;
using namespace luminous;
int main(int argc, char *argv[]) {
    logging::set_log_level(spdlog::level::info);
    Context context{argc, argv};
    context.try_print_help_and_exit();

    Parser sp(&context);
    try {
        if (context.has_scene()) {
            sp.load_from_json(context.scene_file());
        }
    } catch (std::exception &e1) {
        cout << e1.what();
    }

    App app("luminous-gui", luminous::make_int2(480,320));
    return app.run();
}