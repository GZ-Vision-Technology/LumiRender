//
// Created by Zero on 2021/2/20.
//


#pragma once

#include <glad.h>
#include <GLFW/glfw3.h>
#include "core/logging.h"


#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK(call)                                                   \
        do                                                                     \
        {                                                                      \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << "): " << #call         \
                   << std::endl;                                               \
                std::cerr << ss.str() << std::endl;                            \
                throw Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)


#    define GL_CHECK_ERRORS()                                                 \
        do                                                                     \
        {                                                                      \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " << getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << ")";                   \
                std::cerr << ss.str() << std::endl;                            \
                throw Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif

namespace luminous {

    class Exception : public std::runtime_error {
    public:
        Exception(const char *msg)
                : std::runtime_error(msg) {}


    };

    inline const char *getGLErrorString(GLenum error) {
        switch (error) {
            case GL_NO_ERROR:
                return "No error";
            case GL_INVALID_ENUM:
                return "Invalid enum";
            case GL_INVALID_VALUE:
                return "Invalid value";
            case GL_INVALID_OPERATION:
                return "Invalid operation";
                //case GL_STACK_OVERFLOW:      return "Stack overflow";
                //case GL_STACK_UNDERFLOW:     return "Stack underflow";
            case GL_OUT_OF_MEMORY:
                return "Out of memory";
                //case GL_TABLE_TOO_LARGE:     return "Table too large";
            default:
                return "Unknown GL error";
        }
    }

    GLuint createGLShader(const std::string &source, GLuint shader_type) {
        GLuint shader = glCreateShader(shader_type);
        {
            const GLchar *source_data = reinterpret_cast<const GLchar *>( source.data());
            glShaderSource(shader, 1, &source_data, nullptr);
            glCompileShader(shader);

            GLint is_compiled = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
            if (is_compiled == GL_FALSE) {
                GLint max_length = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

                std::string info_log(max_length, '\0');
                GLchar *info_log_data = reinterpret_cast<GLchar *>( &info_log[0]);
                glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

                glDeleteShader(shader);
                std::cerr << "Compilation of shader failed: " << info_log << std::endl;

                return 0;
            }
        }

        GL_CHECK_ERRORS();

        return shader;
    }

    GLuint createGLProgram(const std::string &vert_source, const std::string &frag_source) {
        GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
        if (vert_shader == 0)
            return 0;

        GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
        if (frag_shader == 0) {
            glDeleteShader(vert_shader);
            return 0;
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);

        GLint is_linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
        if (is_linked == GL_FALSE) {
            GLint max_length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log(max_length, '\0');
            GLchar *info_log_data = reinterpret_cast<GLchar *>( &info_log[0]);
            glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
            std::cerr << "Linking of program failed: " << info_log << std::endl;

            glDeleteProgram(program);
            glDeleteShader(vert_shader);
            glDeleteShader(frag_shader);

            return 0;
        }

        glDetachShader(program, vert_shader);
        glDetachShader(program, frag_shader);

        GL_CHECK_ERRORS();

        return program;
    }


    GLint getGLUniformLocation(GLuint program, const std::string &name) {
        GLint loc = glGetUniformLocation(program, name.c_str());
        assert(loc != -1);
        return loc;
    }

    const std::string s_vert_source = R"(
    #version 330 core

    layout(location = 0) in vec3 vertexPosition_modelspace;
    out vec2 UV;

    void main() {
        gl_Position =  vec4(vertexPosition_modelspace,1);
        UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
    }
    )";

    const std::string s_frag_source = R"(
    #version 330 core

    in vec2 UV;
    out vec3 color;

    uniform sampler2D render_tex;
    uniform bool correct_gamma;

    void main() {
        vec2 uv = UV;
        uv.y = 1 - uv.y;
        color = texture( render_tex, uv ).xyz;
        //color = vec3(UV,0);
    }
    )";

    static const GLfloat vertex_buffer_data[] = {
            -1.0f, -1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
            1.0f,  1.0f, 0.0f,
    };
}