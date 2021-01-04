
file(GLOB SOURCE_FILES *.h *.c*)
luminous_render_add_backend(cuda SOURCES ${SOURCE_FILES})

set_target_properties(luminous-backend-cuda PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)