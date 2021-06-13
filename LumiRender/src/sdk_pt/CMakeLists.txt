
cuda_compile_and_embed(sdk_ptx ${PROJECT_SOURCE_DIR}/src/sdk_pt/optixPathTracer.cu)



luminous_render_add_application(sdk-pt SOURCES optixPathTracer.cpp ${sdk_ptx})
