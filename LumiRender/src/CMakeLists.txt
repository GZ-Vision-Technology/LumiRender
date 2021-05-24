include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/stblib/stb")
include_directories("${PROJECT_SOURCE_DIR}/src/ext")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui/glad")

include_directories("${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cxxopts/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/assimp/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/embree/include")

include_directories("${CMAKE_BINARY_DIR}/thirdparty/assimp/include")
include_directories("${CMAKE_BINARY_DIR}/thirdparty/embree/include")

if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_library(luminous-all INTERFACE)
    target_link_libraries(luminous-all INTERFACE
            luminous-core
            luminous-ext
            luminous-gpu
            luminous-cpu
            luminous-render
            luminous-graphics
            luminous-view
            luminous-util)
else()
    add_library(luminous-all INTERFACE)
    target_link_libraries(luminous-all INTERFACE
            luminous-core
            luminous-ext
            luminous-cpu
            luminous-render
            luminous-graphics
            luminous-view
            luminous-util)
endif ()



add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(ext)
if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_definitions (-D CUDA_SUPPORT)
    add_subdirectory(gpu)
endif ()
add_subdirectory(cpu)
add_subdirectory(view)
add_subdirectory(graphics)
add_subdirectory(tests)
add_subdirectory(apps)
add_subdirectory(render)