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



add_subdirectory(core)
add_subdirectory(util)