
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/glad)

add_executable(test_imgui ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(test_imgui PUBLIC glfw_config)

add_library(imgui STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(imgui PUBLIC glfw_config)
set_target_properties(imgui PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)