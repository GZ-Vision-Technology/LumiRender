add_library(luminous-backends INTERFACE)

include_directories(${OptiX_INCLUDE})


#add_subdirectory(framework)
cuda_compile_and_embed(optix_shader_code ${PROJECT_SOURCE_DIR}/src/gpu/shaders/optix_kernels.cu)

file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.cpp)

add_library(luminous-gpu SHARED ${HEADER_FILES} ${SOURCE_FILES} ${optix_shader_code})

target_link_libraries(luminous-gpu
        PUBLIC CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::cudart
        luminous-graphics
        luminous-core
        luminous-render
        )

set_target_properties(luminous-gpu PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

find_package(CUDAToolkit REQUIRED)