add_library(luminous-backends INTERFACE)

include_directories(${OptiX_INCLUDE})


cuda_compile_and_embed(megakernel_pt ${PROJECT_SOURCE_DIR}/src/gpu/shaders/megakernel_pt.cu)

cuda_compile_and_embed(intersect_shader ${PROJECT_SOURCE_DIR}/src/gpu/shaders/intersect.cu)

file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.cpp)

add_library(luminous-gpu SHARED
        ${HEADER_FILES}
        ${SOURCE_FILES}
        ${megakernel_pt}
        ${intersect_shader})

target_link_libraries(luminous-gpu
        PUBLIC CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::cudart
        luminous-base_libs
        luminous-core
        luminous-util)

set_target_properties(luminous-gpu PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

find_package(CUDAToolkit REQUIRED)