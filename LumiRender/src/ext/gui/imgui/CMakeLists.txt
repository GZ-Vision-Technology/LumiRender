
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/glad)

add_executable(test_gui ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(test_gui PUBLIC glfw_config)

add_library(luminous-ext-gui STATIC ${HEADER_FILES} ${SOURCE_FILES})

if(UNIX)
  target_compile_options(luminous-ext-gui PRIVATE "-fPIC")
endif()

target_link_libraries(luminous-ext-gui PUBLIC glfw_config)
set_target_properties(luminous-ext-gui PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)