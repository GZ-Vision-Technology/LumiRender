#include "base_libs/common.h"
#include "gpu/framework/helper/cuda.h"
#include "util/image.h"
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include "rg32f_to_rgb888_ispc.h"

#define NUM_SAMPLES 1024

extern "C" const char spec_albedo_kernels[];

static int spec_image_width, spec_image_height;
static std::string albedo_image_path;
static std::string albedo_avg_image_path;

void usage(const cxxopts::Options &opts) {
    printf(opts.help().c_str());
    exit(-1);
}

void validate_args(int argc, char **argv) {

    try {

        cxxopts::Options opts(argv[0], "Generate cloth specular albedo images and average data set");

        opts.add_options(
                "",
                {{"dimx", "Specular image width", cxxopts::value<int>()->default_value("512")},
                 {"dimy", "Specular albedo image height", cxxopts::value<int>()->default_value("512")},
                 {"o,output", "Specular albedo image file output path", cxxopts::value<std::string>()->default_value("cloth_specular_albedo.jpg")},
                 {"d,avg-output", "Specular albedo average image file output path", cxxopts::value<std::string>()->default_value("cloth_specluar_albedo_average.jpg")},
                 {"h,help", "Print this help message"}});

        auto result = opts.parse(argc, argv);
        if (result.count("help"))
            usage(opts);

        spec_image_width = result["dimx"].as<int>();
        spec_image_height = result["dimy"].as<int>();
        albedo_image_path = result["output"].as<std::string>();
        albedo_avg_image_path = result["avg-output"].as<std::string>();

    } catch (std::exception &e) {
        printf("Parse command line error: %s\n", e.what());
        exit(-1);
    }
}

int write_image_file(const char *fpath, int width, int height, int channel, const void *data) {

    char ext[_MAX_EXT];
    _splitpath(fpath, NULL, NULL, NULL, ext);
    if(_stricmp(ext, ".jpg") == 0 || _stricmp(ext, ".jpeg") == 0)
         if(!stbi_write_jpg(fpath, width, height, channel, data, 9)) {
             fprintf(stderr, "Export \"%s\" error: %s\n", fpath, stbi_failure_reason());
             return -1;
         }
    else if(_stricmp(ext, ".png") == 0)
        if(!stbi_write_png(fpath, width, height, channel, data, channel)) {
            fprintf(stderr, "Export \"%s\" error: %s\n", fpath, stbi_failure_reason());
            return -1;
        }
    else {
        fprintf(stderr, "Unsupported exported file format\n");
        return -1;
    }

    return 0;
}


int main(int argc, char **argv) {

    validate_args(argc, argv);

    CUDA_CHECK(cudaFree(nullptr));

    cudaStream_t cu_sync_strm = 0;
    CUDA_CHECK(cuStreamCreate(&cu_sync_strm, CU_STREAM_DEFAULT));

    CUmodule cu_module;
    CUfunction gen_spec_albedo_image_kernel;
    CUfunction gen_spec_albedo_avg_kernel;
    cudaEvent_t gen_albedo_done_ev, gen_albedo_avg_done_ev;

    CUDA_CHECK(cuModuleLoadData(&cu_module, spec_albedo_kernels));
    CUDA_CHECK(cuModuleGetFunction(&gen_spec_albedo_image_kernel, cu_module, "cloth_spec_albedo_kernel"));
    CUDA_CHECK(cuModuleGetFunction(&gen_spec_albedo_avg_kernel, cu_module, "cloth_spec_albedo_avg_kernel"));

    CUDA_CHECK(cuEventCreate(&gen_albedo_done_ev, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&gen_albedo_avg_done_ev, CU_EVENT_DEFAULT));

    const size_t spitch = spec_image_width * sizeof(luminous::float2);

    // Specular albedo image resource
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t spec_albedo_buffer;
    CUDA_CHECK(cudaMallocArray(&spec_albedo_buffer, &channelDesc, spec_image_width, spec_image_height, cudaArraySurfaceLoadStore));
    cudaResourceDesc resDesc;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = spec_albedo_buffer;
    cudaSurfaceObject_t spec_albedo_image;
    CUDA_CHECK(cudaCreateSurfaceObject(&spec_albedo_image, &resDesc));

    // Specular albedo average image resource
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t spec_albedo_avg_buffer;
    CUDA_CHECK(cudaMallocArray(&spec_albedo_avg_buffer, &channelDesc, spec_image_height, 1, cudaArraySurfaceLoadStore));
    resDesc.res.array.array = spec_albedo_avg_buffer;
    cudaSurfaceObject_t spec_albedo_avg_image;
    CUDA_CHECK(cudaCreateSurfaceObject(&spec_albedo_avg_image, &resDesc));

    int nsample = NUM_SAMPLES;
    uint2 image_dim = make_uint2(spec_image_width, spec_image_height);

    void *spec_albedo_args[] = {
            &nsample,
            &image_dim,
            &spec_albedo_image};

    CUDA_CHECK(cuLaunchKernel(gen_spec_albedo_image_kernel, (spec_image_width + 31) >> 5, (spec_image_height + 31) >> 5,
                              1, 32, 32, 1, 0, cu_sync_strm, spec_albedo_args, nullptr));
    CUDA_CHECK(cuEventRecord(gen_albedo_done_ev, cu_sync_strm));

    void *spec_avg_args[] = {
            &spec_albedo_image,
            &image_dim,
            &spec_albedo_avg_image};

    CUDA_CHECK(cuStreamWaitEvent(cu_sync_strm, gen_albedo_done_ev, CU_EVENT_WAIT_DEFAULT));
    CUDA_CHECK(cuLaunchKernel(gen_spec_albedo_avg_kernel, (spec_image_height + 31) >> 5, 1, 1, 32, 1, 1, 0, cu_sync_strm, spec_avg_args,
                              nullptr));
    CUDA_CHECK(cuEventRecord(gen_albedo_avg_done_ev, cu_sync_strm));

    // Copy back to CPU side
    std::vector<luminous::float2> cb_buffer{(size_t)spec_image_width * spec_image_height};

    CUDA_CHECK(cuStreamWaitEvent(cu_sync_strm, gen_albedo_done_ev, CU_EVENT_WAIT_DEFAULT));
    CUDA_CHECK(cudaMemcpy2DFromArrayAsync(cb_buffer.data(), spitch, spec_albedo_buffer,
                                          0, 0, spitch, spec_image_height,
                                          cudaMemcpyDeviceToHost, cu_sync_strm));
    CUDA_CHECK(cuStreamSynchronize(cu_sync_strm));

    ispc::rg32f_to_rgb888(spec_image_width * spec_image_height * 2, (const float *) cb_buffer.data(), (uint8_t *) cb_buffer.data());

    if(write_image_file(albedo_image_path.c_str(), spec_image_width, spec_image_height, 3, cb_buffer.data()))
        return -1;

    CUDA_CHECK(cuStreamWaitEvent(cu_sync_strm, gen_albedo_avg_done_ev, CU_EVENT_WAIT_DEFAULT));
    CUDA_CHECK(cudaMemcpyFromArrayAsync(cb_buffer.data(), spec_albedo_avg_buffer, 0, 0, spec_image_height * sizeof(luminous::float2),
                                cudaMemcpyDeviceToHost, cu_sync_strm));
    CUDA_CHECK(cuStreamSynchronize(cu_sync_strm));

    ispc::rg32f_to_rgb888(spec_image_height * 2, (const float *) cb_buffer.data(), (uint8_t *) cb_buffer.data());

    if(write_image_file(albedo_avg_image_path.c_str(), spec_image_height, 1, 3, cb_buffer.data()))
        return -1;

    CUDA_CHECK(cudaDestroySurfaceObject(spec_albedo_avg_image));
    CUDA_CHECK(cudaFreeArray(spec_albedo_avg_buffer));
    CUDA_CHECK(cudaDestroySurfaceObject(spec_albedo_image));
    CUDA_CHECK(cudaFreeArray(spec_albedo_buffer));

    CUDA_CHECK(cuEventDestroy(gen_albedo_done_ev));
    CUDA_CHECK(cuEventDestroy(gen_albedo_avg_done_ev));
    CUDA_CHECK(cuModuleUnload(cu_module));

    CUDA_CHECK(cuStreamDestroy(cu_sync_strm));

    return 0;
}