luminous_render_add_application(test-cuda SOURCES math_cuda.cpp)
#target_link_libraries(test-cuda PRIVATE CUDA::nvrtc CUDA::cuda_driver)