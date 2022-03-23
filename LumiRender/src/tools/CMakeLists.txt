


cuda_compile_and_embed(spec_albedo_kernels ${CMAKE_CURRENT_SOURCE_DIR}/cloth_spec/spec_albedo.cu)

add_executable(
  cloth_spec
  cloth_spec/cloth_spec.cpp
  ${spec_albedo_kernels}
  cloth_spec/rg32f_to_rgb888.ispc
)

target_link_libraries(
  cloth_spec
  PRIVATE
  CUDA::nvrtc
  CUDA::cudart
  luminous-util
)