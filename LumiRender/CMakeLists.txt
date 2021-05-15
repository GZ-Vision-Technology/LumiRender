cmake_minimum_required(VERSION 3.17)
project(LumiRender)

# 项目信息
set(APP_NAME LumiRender CACHE STRING INTERNAL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

include(configure_optix)

set(CMAKE_CXX_STANDARD 11)
include(CheckLanguage)

function(luminous_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luminous-all)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")
endfunction()


#add_subdirectory(thirdparty/spdlog)
#add_library(spdlog_config INTERFACE)
#target_link_libraries(spdlog_config INTERFACE spdlog::spdlog)
#
#add_subdirectory(thirdparty/cxxopts)
#add_library(cxxopts_config INTERFACE)
#target_link_libraries(cxxopts_config INTERFACE cxxopts::cxxopts)

add_subdirectory(thirdparty/assimp)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp)
set_target_properties(assimp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")

set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
#set_property(EMBREE_TUTORIALS OFF)

add_subdirectory(thirdparty/embree)
add_library(embree_config INTERFACE)
target_link_libraries(embree_config INTERFACE embree::embree)
set_target_properties(embree PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")

set(CMAKE_DEBUG_POSTFIX "")

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions (-D DEBUG_BUILD)
endif ()

set(CMAKE_CXX_STANDARD 17)
add_definitions(-D _ENABLE_EXTENDED_ALIGNED_STORAGE)
add_subdirectory(src)

set (CHECK_CUDA_OUTPUT_EXE ${CMAKE_BINARY_DIR}/check_cuda)
if (MSVC)
    execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/check_cuda.cu -ccbin ${CMAKE_CXX_COMPILER} -o ${CHECK_CUDA_OUTPUT_EXE})
else  ()
    execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/check_cuda.cu -o ${CHECK_CUDA_OUTPUT_EXE})
endif ()

execute_process (COMMAND ${CHECK_CUDA_OUTPUT_EXE}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE CHECK_CUDA_OUTPUT)

message(${CHECK_CUDA_OUTPUT_EXE})
message(${CUDA_RETURN_CODE})
message(${CHECK_CUDA_OUTPUT})

if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
    message (SEND_ERROR ${CHECK_CUDA_OUTPUT})
else ()
    set(ARCH "${CHECK_CUDA_OUTPUT}")
    message (STATUS "Detected CUDA Architecture: ${ARCH}")
    string (APPEND CMAKE_CUDA_FLAGS " --gpu-architecture=${ARCH}")
endif ()
