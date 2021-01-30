add_subdirectory(glfw)
add_library(glfw_config INTERFACE)
target_link_libraries(glfw_config INTERFACE glfw)

add_subdirectory(imgui)