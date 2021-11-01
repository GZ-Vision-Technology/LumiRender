cmake_minimum_required(VERSION 3.17)
project(LumiRender)

if (WIN32)
    if(NOT DEFINED ENV{VLD_SDK_PATH})
        message(FATAL_ERROR "Can not find VLD SDK $ENV{VLD_SDK_PATH} in build environment!")
    else()
        message(STATUS "VLD SDK found in '$ENV{VLD_SDK_PATH}'")
    endif()

    include_directories($ENV{VLD_SDK_PATH}/include)
    link_directories($ENV{VLD_SDK_PATH}/lib/Win64)
endif(WIN32)


# 项目信息
set(APP_NAME LumiRender CACHE STRING INTERNAL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    include(configure_optix)
    include(CheckLanguage)
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT ON)
else ()
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT OFF)
    message(not find cuda !!)
endif ()

set(CMAKE_CXX_STANDARD 11)

function(luminous_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luminous-all)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")
endfunction()

#add_subdirectory(thirdparty/assimp)
#add_library(assimp_config INTERFACE)
#target_link_libraries(assimp_config INTERFACE assimp::assimp)
#set_target_properties(assimp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")

#set(LUMINOUS_VCPKG_PATH "E:/work/tools/vcpkg/installed/x64-windows/include")

find_package(assimp CONFIG REQUIRED)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp)

set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)

set(CMAKE_DEBUG_POSTFIX "")

set(LUMINOUS_CPU_BACKEND ON)
set(LUMINOUS_LOCAL_EMBREE ON)

if (LUMINOUS_LOCAL_EMBREE)
    find_package(embree 3 CONFIG REQUIRED)
else()
#    add_subdirectory(thirdparty/embree)
#    add_library(embree_config INTERFACE)
#    target_link_libraries(embree_config INTERFACE embree3)
#    set_target_properties(embree PROPERTIES
#            PUBLIC_HEADER "${HEADER_FILES}"
#            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

find_package(OpenImageIO REQUIRED)

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

if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
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
endif ()


