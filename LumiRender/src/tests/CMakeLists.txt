
if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    cuda_compile_and_embed(ptxCode ${PROJECT_SOURCE_DIR}/src/gpu/shaders/test_kernels.cu)

    luminous_render_add_application(test-kernel_launch SOURCES test_kernel_launch.cpp ${ptxCode})
    target_link_libraries(test-kernel_launch PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)

    luminous_render_add_application(test-cuda_tex_test SOURCES cuda_tex_test.cpp ${ptxCode})
    target_link_libraries(test-cuda_tex_test PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)
endif ()


luminous_render_add_application(test-graphicslib SOURCES test_graphics.cpp)

luminous_render_add_application(test-image-io SOURCES test_image_io.cpp)
luminous_render_add_application(test-sampler SOURCES test_sampler.cpp)
luminous_render_add_application(test-camera SOURCES test_camera.cpp)
luminous_render_add_application(test-distribution2d SOURCES distribution2d_test.cpp)
luminous_render_add_application(test-variant SOURCES test_variant.cpp)
luminous_render_add_application(test-embree SOURCES test_embree.cpp)
luminous_render_add_application(test-soa SOURCES test_soa.cpp)
luminous_render_add_application(test-parallel SOURCES test_parallel.cpp)
luminous_render_add_application(test-arena SOURCES test_arena.cpp)
luminous_render_add_application(test-reflection SOURCES test_reflection.cpp)
luminous_render_add_application(test-upload SOURCES test_upload.cpp)
luminous_render_add_application(test_tagged_ptr SOURCES test_tagged_ptr.cpp)

add_executable(playground playground.cpp)

#set(CMAKE_CUDA_STANDARD 17)
#project(cuda_test CUDA)
#
#luminous_render_add_application(cuda_test SOURCES test_cuda.cu)
#
#set_target_properties(
#        cuda_test
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
