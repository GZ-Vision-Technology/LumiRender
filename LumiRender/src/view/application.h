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
#include <glad.h>
#include "util/clock.h"
#include "render/include/task.h"

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
        int2 _last_mouse_pos = make_int2(0);

        bool _is_gpu_rendering = false;

        unique_ptr<Task> _task;

        bool _show_window{false};

        bool _left_key_press{false};

        bool _right_key_press{false};

        bool _need_update{true};

        HpClock _clock;

        struct FrameStats {
            float update_time{.0f};
            float render_time{.0f};
            float display_time{.0f};
            float last_frame_elapsed{.0f};
            float last_sample_elapsed{.0f};
            unsigned long frame_count = 0;
        } _frame_stats;

        void *_render_buffer_shared_resource{};

    public:
        App() = default;

        App(const std::string &title, const int2 &size, Context *context, const Parser &parser);

        ~App();

        void init(const std::string &title, const int2 &size, Context *context, const Parser &parser);

        int run();

    private:
        void init_with_gui(const std::string &title);

        void init_gl_context();

        void init_window(const std::string &title, const uint2 &size);

        void init_imgui();

        void on_resize(const uint2 &new_size);

        void on_key_event(int key, int scancode, int action, int mods);

        void on_cursor_move(int2 new_pos);

        void on_mouse_event(int button, int action, int mods);

        void on_scroll_event(double scroll_x, double scroll_y);

        void set_title(const std::string &s);

        void render(double delta_elapsed);

        void check_and_update();

        void draw() const;

        void imgui_begin();

        void imgui_end();

        void update_pixel_buffer();

        void display_stats();

        int run_with_gui();

        int run_with_cli();
    };
}