//
// Created by Zero on 2021/1/29.
//

#include <iostream>
#include "core/context.h"
#include "parser/json_parser.h"
#include "parser/assimp_parser.h"
#include <memory>
#include "view/application.h"
#include "core/platform.h"

using std::cout;
using std::endl;
using namespace luminous;


static void print_platform_info() {

    auto print_isa_info_avx2 = [] {
        logging::info("ISPC acceleration will use x86-64:AVX2 instruction set");
    };
    auto print_isa_info_avx = [] {
        logging::info("ISPC acceleration will use x86-64:AVX instruction set");
    };
    auto print_isa_info_sse4 = [] {
        logging::info("ISPC acceleration will use x86-64:SSE41 instruction set");
    };
    auto print_isa_info_sse2 = [] {
        logging::info("ISPC acceleration will use x86-64:SSE2 instruction set");
    };

    CALL_ISPC_ROUTINE_BY_HARDWARE_FEATURE(print_isa_info) {
        logging::info("ISA not found");
    }
}



int execute(int argc, char *argv[]) {
    logging::set_log_level(spdlog::level::info);

    print_platform_info();

    Context context{argc, argv};
    context.try_print_help_and_exit();
    if (argc == 1) {
        context.print_help();
        return 0;
    }
    std::unique_ptr<Parser> parser{nullptr};
    App app;
    try {
        set_thread_num(context.thread_num());
        if (context.has_scene()) {
            auto scene_file = context.scene_file();
            if (scene_file.extension() == ".json" || scene_file.extension() == ".bson") {
                parser = std::make_unique<JsonParser>(&context);
            } else {
                parser = std::make_unique<AssimpParser>(&context);
            }
            parser->load(scene_file);
        }
        app.init("luminous", luminous::make_int2(1280,720), &context, *parser);
    } catch (std::exception &e1) {
        cout << "scene load exception : " << e1.what() << endl;
        return 1;
    }
    return app.run();
}

int main(int argc, char *argv[]) {
    auto ret = execute(argc, argv);
    return ret;
}