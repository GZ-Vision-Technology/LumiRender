file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)


include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(wavefront_kernels ${PROJECT_SOURCE_DIR}/src/render/integrators/wavefront/kernels.cu)

add_library(luminous-render SHARED ${HEADER_FILES} ${SOURCE_FILES} ${wavefront_kernels})


target_link_libraries(luminous-render
        PUBLIC
        embree
#        OpenImageIO
        luminous-base_libs
        luminous-util
        luminous-denoise
        luminous-cpu
        luminous-gpu
        luminous-core)

set_target_properties(luminous-render PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)