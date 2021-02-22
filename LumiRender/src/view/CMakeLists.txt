file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(luminous-view SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(luminous-view
        PUBLIC
        luminous-ext
        luminous-util
        luminous-gpu
        luminous-core)

set_target_properties(luminous-view PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)