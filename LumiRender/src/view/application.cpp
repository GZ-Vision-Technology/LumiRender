//
// Created by Zero on 2021/2/19.
//

#include <glad.h>
#include "application.h"

namespace luminous {
    App::App(const std::string &title, const int2 &size) {
        init_window(title, size);
    }

    void App::init_window(const std::string &title, const int2 &size) {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            exit(EXIT_FAILURE);

        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, true);
        _handle = glfwCreateWindow(size.x, size.y, title.c_str(), NULL, NULL);
        if (!_handle) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        gladLoadGL();

        glfwSetWindowUserPointer(_handle, this);
        glfwMakeContextCurrent(_handle);
        glfwSwapInterval( 1 );
    }

    void App::set_title(const std::string &s) {
        glfwSetWindowTitle(_handle,s.c_str());
    }

    void App::resize(const int2 &new_size) {

    }


}