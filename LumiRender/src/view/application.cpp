//
// Created by Zero on 2021/2/19.
//


#include "application.h"
#include <iostream>
#include "util/image_io.h"

using namespace std;

namespace luminous {

    static float4 bg_color = make_float4(0.2);

    template<typename T = App>
    T *get_user_ptr(GLFWwindow *window) {
        return static_cast<T *>(glfwGetWindowUserPointer(window));
    }

    static void on_glfw_error(int error, const char *description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    }

    /*! callback for a window resizing event */
    static void glfw_resize(GLFWwindow *window, int width, int height) {
        get_user_ptr(window)->on_resize(make_int2(width, height));
    }

    /*! callback for a char key press or release */
    static void glfw_char_event(GLFWwindow *window,
                                unsigned int key) {

    }

    /*! callback for a key press or release*/
    static void glfw_key_event(GLFWwindow *window,
                               int key,
                               int scancode,
                               int action,
                               int mods) {
        get_user_ptr(window)->on_key_event(key, scancode, action, mods);
    }

    /*! callback for _moving_ the mouse to a new position */
    static void glfw_cursor_move(GLFWwindow *window, double x, double y) {
        get_user_ptr(window)->on_cursor_move(make_int2(x, y));
    }

    /*! callback for pressing _or_ releasing a mouse button*/
    static void glfw_mouse_event(GLFWwindow *window,
                                 int button,
                                 int action,
                                 int mods) {
        get_user_ptr(window)->on_mouse_event(button, action, mods);
    }

    static void glfw_scroll_event(GLFWwindow *window, double scroll_x, double scroll_y) {
        get_user_ptr(window)->on_scroll_event(scroll_x, scroll_y);
    }

    void App::on_cursor_move(int2 new_pos) {
        int2 delta = new_pos - _last_mouse_pos;
        if (!_last_mouse_pos.is_zero()) {
            _task->update_camera_view(delta.x, -delta.y);
        }
        _last_mouse_pos = new_pos;
    }

    void App::on_scroll_event(double scroll_x, double scroll_y) {
        _task->update_camera_fov_y(scroll_y);
    }

    void App::on_key_event(int key, int scancode, int action, int mods) {
        _task->on_key(key, scancode, action, mods);
    }

    void App::on_mouse_event(int button, int action, int mods) {
        // todo
    }

    void App::on_resize(const int2 &new_size) {
        glViewport(0, 0, new_size.x, new_size.y);
        _task->update_film_resolution(new_size);
    }

    App::App(const std::string &title, const int2 &size, Context *context, const Parser &parser)
            : _size(size) {
        _task = make_unique<CUDATask>(context);
        _task->init(parser);
        init_window(title, _task->resolution());
        init_event_cb();
        init_imgui();
        init_gl_context();
    }

    void App::init_gl_context() {
        auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\HelloWorld.png)";
        auto[rgb, res] = load_image(path);
        test_color = new uint32_t[res.y * res.x];
        for (int i = 0; i < res.y * res.x; ++i) {
            test_color[i] = make_rgba(rgb[i]);
        }

        glGenTextures(1, &_gl_ctx.fb_texture);
        glBindTexture(GL_TEXTURE_2D, _gl_ctx.fb_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.x, res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, test_color);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                        GL_REPEAT);    // set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        _gl_ctx.program = createGLProgram(s_vert_source, s_frag_source);
        _gl_ctx.program_tex = getGLUniformLocation(_gl_ctx.program, "render_tex");
        glUseProgram(_gl_ctx.program);
        glUniform1i(_gl_ctx.program_tex, 0);

        glBindTexture(GL_TEXTURE_2D, 0);

        glGenVertexArrays(1, &_gl_ctx.vao);
        glGenBuffers(1, &_gl_ctx.vbo);

        glBindVertexArray(_gl_ctx.vao);

        glBindBuffer(GL_ARRAY_BUFFER, _gl_ctx.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void App::imgui_begin() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Render();
    }

    void App::imgui_end() {
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    int App::run() {
        while (!glfwWindowShouldClose(_handle)) {
            loop();
            imgui_begin();
            glClearColor(bg_color.x, bg_color.y, bg_color.z, bg_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            render();
            update_render_texture();
            draw();
            imgui_end();
            glfwPollEvents();
            glfwSwapBuffers(_handle);
        }
        return 0;
    }

    void App::init_event_cb() {
        glfwSetFramebufferSizeCallback(_handle, glfw_resize);
        glfwSetMouseButtonCallback(_handle, glfw_mouse_event);
        glfwSetKeyCallback(_handle, glfw_key_event);
        glfwSetCharCallback(_handle, glfw_char_event);
        glfwSetCursorPosCallback(_handle, glfw_cursor_move);
        glfwSetScrollCallback(_handle, glfw_scroll_event);
    }

    void App::draw() const {
        glUniform1i(_gl_ctx.program_tex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _gl_ctx.fb_texture);
        glUseProgram(_gl_ctx.program);
        glBindVertexArray(_gl_ctx.vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void App::render() {
        _task->render_gui();
    }

    void App::init_window(const std::string &title, const int2 &size) {
        glfwSetErrorCallback(on_glfw_error);
        if (!glfwInit())
            exit(EXIT_FAILURE);

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        _handle = glfwCreateWindow(size.x, size.y, title.c_str(), NULL, NULL);
        if (!_handle) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwSetWindowUserPointer(_handle, this);
        glfwMakeContextCurrent(_handle);
        glfwSwapInterval(1);

        if (gladLoadGL() == 0) {
            fprintf(stderr, "Failed to initialize OpenGL loader!\n");
            exit(EXIT_FAILURE);
        }
    }

    void App::init_imgui() {
        const char *glsl_version = "#version 130";
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(_handle, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    void App::set_title(const std::string &s) {
        glfwSetWindowTitle(_handle, s.c_str());
    }

    void App::update_render_texture() {
        auto path = R"(E:\work\graphic\renderer\LumiRender\LumiRender\res\image\HelloWorld.png)";
        auto[rgb, res] = load_image(path);
        test_color = new uint32_t[res.y * res.x];
        for (int i = 0; i < res.y * res.x; ++i) {
            test_color[i] = make_rgba(rgb[i]);
        }

        glBindTexture(GL_TEXTURE_2D, _gl_ctx.fb_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.x, res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, test_color);
    }

}