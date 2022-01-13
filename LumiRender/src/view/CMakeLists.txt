project(liminous-view LANGUAGES CXX)

set(HEADER_FILES
  application.h
  gl_helper.h
)

set(SOURCE_FILES
  application.cpp
)

add_library(luminous-view SHARED ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(luminous-view PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
)

target_compile_definitions(luminous-view PRIVATE LUMINOUS_VIEW_BUILD)

target_include_directories(luminous-view INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(luminous-view
        PUBLIC
        luminous-ext
        luminous-util
        luminous-render
        luminous-cpu
        luminous-core)

set_target_properties(luminous-view PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")