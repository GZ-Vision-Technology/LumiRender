//
// Created by Zero on 2021/2/19.
//


#include "application.h"


namespace luminous {
    App::App(const std::string &title, const int2 &size) {
        init_window(title, size);
    }

    int App::run() {
        while (!glfwWindowShouldClose(_handle)) {
            loop();
            glfwPollEvents();
            int display_w, display_h;
            glfwGetFramebufferSize(_handle, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(1,0,0,0);
            glClear(GL_COLOR_BUFFER_BIT);
            glfwSwapBuffers(_handle);
        }
        return 0;
    }

    void App::init_window(const std::string &title, const int2 &size) {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            exit(EXIT_FAILURE);

        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        _handle = glfwCreateWindow(size.x, size.y, title.c_str(), NULL, NULL);
        if (!_handle) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(_handle);
        glfwSwapInterval( 1 );

        if (gladLoadGL() == 0) {
            fprintf(stderr, "Failed to initialize OpenGL loader!\n");
            exit(EXIT_FAILURE);
        }
    }

    void App::set_title(const std::string &s) {
        glfwSetWindowTitle(_handle,s.c_str());
    }

    void App::resize(const int2 &new_size) {

    }

}