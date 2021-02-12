
file(GLOB SOURCE_FILES *.h *.c*)


luminous_render_add_backend(cuda SOURCES ${SOURCE_FILES})
target_include_directories(luminous-backend-cuda PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

target_link_libraries(luminous-backend-cuda PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)

set_target_properties(luminous-backend-cuda PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)