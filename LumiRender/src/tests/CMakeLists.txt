luminous_render_add_application(test-kernel_launch SOURCES test_kernel_launch.cpp)
target_link_libraries(test-kernel_launch PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)

luminous_render_add_application(test-graphicslib SOURCES test_graphics.cpp)

set(CMAKE_CUDA_STANDARD 17)
project(cuda_test CUDA)

luminous_render_add_application(cuda_test SOURCES test_cuda.cu)

set_target_properties(
        cuda_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
