

add_subdirectory(stblib)
add_subdirectory(tinyexr)
add_subdirectory(xatlas)

add_library(luminous-ext INTERFACE)
target_link_libraries(luminous-ext INTERFACE
        luminous-ext-xatlas
        luminous-ext-stb
        luminous-ext-tinyexr)