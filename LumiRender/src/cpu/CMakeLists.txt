
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)


add_library(luminous-cpu SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(luminous-cpu
        PUBLIC
        luminous-base_libs
        embree
        luminous-core
        luminous-util)

set_target_properties(luminous-cpu PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)