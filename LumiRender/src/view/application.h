//
// Created by Zero on 2021/2/19.
//


#pragma once

#include "gui/imgui/imgui.h"
#include "gui/imgui/imgui_impl_glfw.h"
#include "gui/imgui/imgui_impl_opengl3.h"
#include <iostream>
#include "graphics/header.h"
#include "core/logging.h"
#include "graphics/math/common.h"
#include "gl_helper.h"
#include "gpu/framework/cuda_pipeline.h"

namespace luminous {

    class App {
        struct GLContext {
            GLuint program{0};
            GLuint fb_texture{0};
            GLuint program_tex{0};
            GLuint vao{0};
            GLuint vbo{0};
        };
    private:
        int2 _size;
        GLContext _gl_ctx;
        GLFWwindow *_handle{nullptr};
        uint32_t * test_color{};
        int2 _last_mouse_pos = make_int2(-1);

        unique_ptr<Pipeline> _pipeline;

    public:
        App(const std::string &title, const int2 &size);

        void init_pipeline();

        void init_gl_context();

        void init_window(const std::string &title, const int2 &size);

        void init_event_cb();

        void init_imgui();

        void on_resize(const int2 &new_size);

        void on_key_event(int key, int scancode,int action,int mods);

        void on_cursor_move(int2 new_pos);

        void on_mouse_event(int button, int action,int mods);

        void set_title(const std::string &s);

        void render();

        void draw() const;

        void imgui_begin();

        void imgui_end();

        void loop() {
            render();
        }

        int run();
    };
}