project(liminous-view LANGUAGES CXX)

set(HEADER_FILES
  application.h
  gl_helper.h
)

set(SOURCE_FILES
  application.cpp
)

add_library(luminous-view STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(luminous-view INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(luminous-view
        PUBLIC
        luminous-ext
        luminous-util
        luminous-render
        luminous-cpu
        luminous-core)