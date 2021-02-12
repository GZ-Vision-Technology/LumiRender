add_library(luminous-backends INTERFACE)

include_directories(${OptiX_INCLUDE})

function(luminous_render_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luminous-backend-${name} MODULE ${BACKEND_SOURCES})
    target_compile_features(luminous-backend-${name} PRIVATE c_std_11 cxx_std_17)
    add_dependencies(luminous-backends luminous-backend-${name})
    set_target_properties(luminous-backend-${name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/backends/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/backends/")
endfunction()


add_subdirectory(framework)
