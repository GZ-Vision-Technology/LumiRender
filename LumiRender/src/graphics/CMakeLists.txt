

add_subdirectory(lstd)

add_library(luminous-graphics INTERFACE)

target_link_libraries(luminous-graphics INTERFACE
        graphics-lstd)
