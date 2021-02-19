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
#include "gl/GL.h"
#include <GLFW/glfw3.h>

namespace luminous {

    static void glfw_error_callback(int error, const char* description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    }

    class App {
    private:
        int2    _size;
        GLuint   _fb_texture;
        GLFWwindow * _handle { nullptr };
        int2 _last_mouse_pos = make_int2(-1);
    public:
        App(const std::string &title,
            const int2 &size);

        void init_window(const std::string &title, const int2 &size);

        void resize(const int2 &new_size);

        void set_title(const std::string &s);

        void render(){}

        void draw(){}

        void loop() {
            render();
            draw();
        }

        int run() {
            while (!glfwWindowShouldClose(_handle)) {
                loop();
                glfwSwapBuffers(_handle);
                glfwPollEvents();
                int display_w, display_h;
                glfwGetFramebufferSize(_handle, &display_w, &display_h);
                glViewport(0, 0, display_w, display_h);
                glClearColor(0,0,0,0);
                glClear(GL_COLOR_BUFFER_BIT);
            }
            return 0;
        }
    };
}