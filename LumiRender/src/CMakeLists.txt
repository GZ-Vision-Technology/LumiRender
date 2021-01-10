include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/stblib/stb")


include_directories("${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cxxopts/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/assimp/include")

include_directories("${CMAKE_BINARY_DIR}/thirdparty/assimp/include")

add_library(luminous-all INTERFACE)
target_link_libraries(luminous-all INTERFACE
        luminous-core
        luminous-ext
        luminous-util)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT ON)
else ()
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT OFF)
    message(not find cuda !!)
endif ()

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(ext)
add_subdirectory(backends)
#add_subdirectory(ui)
add_subdirectory(tests)