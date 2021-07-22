//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "render/include/parser.h"
#include <memory>
#include "gpu/framework/cuda_task.h"

using namespace luminous;
using std::cout;
int main(int argc, char *argv[]) {
    logging::set_log_level(spdlog::level::info);
    Context context{argc, argv};
    context.try_print_help_and_exit();
    if (!context.has_scene()) {
        context.print_help();
        return 0;
    }
    Parser sp(&context);
    try {
        sp.load_from_json(context.scene_file());
    } catch (std::exception &e1) {
        cout << e1.what();
        context.print_help();
    }
    std::unique_ptr<luminous::Task> task;
    if (context.device() == "cuda") {
        task = std::make_unique<CUDATask>(&context);
    } else if (context.device() == "cpu") {
        LUMINOUS_INFO("cpu is not support");
    }
    task->init(sp);

    return 0;
}