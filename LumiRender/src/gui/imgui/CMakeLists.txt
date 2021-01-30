file(GLOB_RECURSE SRC *.c*)

message(wori)
message(${SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/glad)

add_library(imgui ${SRC})
target_link_libraries(imgui PUBLIC glfw_config)