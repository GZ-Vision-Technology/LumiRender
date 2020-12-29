cmake_minimum_required(VERSION 3.17)
project(LumiRender)

# 项目信息
set(APP_NAME LumiRender CACHE STRING INTERNAL)

set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)

add_subdirectory(thirdparty/spdlog)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog)

add_subdirectory(thirdparty/cxxopts)
add_library(cxxopts_config INTERFACE)
target_link_libraries(cxxopts_config INTERFACE cxxopts::cxxopts)

add_subdirectory(thirdparty/assimp)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp)

set(CMAKE_DEBUG_POSTFIX "")

add_subdirectory(src)