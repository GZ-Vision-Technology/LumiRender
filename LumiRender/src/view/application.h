//
// Created by Zero on 2021/2/19.
//


#pragma once

#include "gui/imgui/imgui.h"
#include "gui/imgui/imgui_impl_glfw.h"
#include "gui/imgui/imgui_impl_opengl3.h"
#include <iostream>
#include "base_libs/header.h"
#include "core/logging.h"
#include "base_libs/math/common.h"
#include "gl_helper.h"
#include "util/clock.h"
#include "render/include/task.h"

namespace luminous {
    using namespace std::chrono;
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
        uint32_t *test_color{};
        int2 _last_mouse_pos = make_int2(0);

        unique_ptr<Task> _task;

        bool _show_window{false};

        Clock _clock;

        bool _left_key_press{false};

        bool _right_key_press{false};

        bool _need_update{true};

        uint test_count = 0;

        double acc_t = 0;

    public:
        App(const std::string &title, const int2 &size, Context *context, const JsonParser &parser);

        void init_with_gui(const std::string &title) {
            init_window(title, _task->resolution());
            init_event_cb();
            init_imgui();
            init_gl_context();
        }

        void init_gl_context();

        void update_render_texture();

        void init_window(const std::string &title, const uint2 &size);

        void init_event_cb();

        void init_imgui();

        void on_resize(const uint2 &new_size);

        void on_key_event(int key, int scancode, int action, int mods);

        void on_cursor_move(int2 new_pos);

        void on_mouse_event(int button, int action, int mods);

        void on_scroll_event(double scroll_x, double scroll_y);

        void set_title(const std::string &s);

        void render();

        void check_and_update();

        void draw() const;

        void imgui_begin();

        void imgui_end();

        int run();

        int run_with_gui();

        int run_with_cli();
    };
}