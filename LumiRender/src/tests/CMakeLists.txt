luminous_render_add_application(math-cuda SOURCES math_cuda.cpp)
target_link_libraries(math-cuda PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)

luminous_render_add_application(test-graphicslib SOURCES test_graphics.cpp)

set(CMAKE_CUDA_STANDARD 14)
project(cuda_test CUDA)

luminous_render_add_application(cuda_test SOURCES test_cuda.cu)

set_target_properties(
        cuda_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
