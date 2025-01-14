include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/stblib/stb")
include_directories("${PROJECT_SOURCE_DIR}/src/ext")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui/glad")

include_directories("${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cxxopts/include")


if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_library(luminous-all INTERFACE)
    if (LUMINOUS_CPU_BACKEND)
        target_link_libraries(luminous-all INTERFACE
                luminous-core
                luminous-ext
                luminous-gpu
                luminous-cpu
                luminous-render
                luminous-denoise
                luminous-base_libs
                luminous-view
                luminous-util)
    else()
        target_link_libraries(luminous-all INTERFACE
                luminous-core
                luminous-ext
                luminous-gpu
                luminous-render
                luminous-base_libs
                luminous-view
                luminous-util)
    endif ()

else()
    add_library(luminous-all INTERFACE)
    target_link_libraries(luminous-all INTERFACE
            luminous-core
            luminous-ext
            luminous-cpu
            luminous-render
            luminous-base_libs
            luminous-view
            luminous-util)
endif ()



add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(parser)
add_subdirectory(ext)
if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_definitions (-DCUDA_SUPPORT)
    add_subdirectory(gpu)
endif ()
if (LUMINOUS_CPU_BACKEND)
    add_subdirectory(cpu)
endif ()
add_subdirectory(view)
add_subdirectory(base_libs)

add_subdirectory(apps)
add_subdirectory(render)
add_subdirectory(denoise)

add_subdirectory(tools)

if(${LUMINOUS_BUILD_TESTS})
    add_subdirectory(tests)
endif()

