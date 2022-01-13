#include <iomanip>
#include <iostream>
#include <optix.h>
#include <optix_denoiser_tiling.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "film_optix_denoiser.h"
#include "core_exception.h"

namespace luminous {

#if OPTIX_VERSION < 70300

class FilmDummyOptixDenoiser: public FilmDenoiser {

public:
    FilmDummyOptixDenoiser() = default;
    int init_device() override {
        assert("OptiX SDK version prior 7.3 is not compatible with this implementation");
        return -1;
    }
    int init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode) override {
        return 0;
    }
    int update(const FilmDenoiserInputData &data) override {
        return 0;
    }
    int exec() override {
        return 0;
    }
    int get_results() override {
        return 0;
    }
};

std::unique_ptr<FilmDenoiser> create_film_optix_denoiser() {
    return std::unique_ptr<FilmDenoiser>{new FilmDummyOptixDenoiser};
}

#else

class FilmOptixDenoiser: public FilmDenoiser {

public:
    FilmOptixDenoiser() = default;

protected:
    int init_device() override;
    int init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode) override;
    int update(const FilmDenoiserInputData &data) override;
    int exec() override;
    int get_results() override;
    ~FilmOptixDenoiser();

private:
    int init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool kpMode, bool temporalMode);

    OptixImage2D conditional_create_optix_image2d(FilmDenoiserBufferView bv, unsigned int width, unsigned int height, bool copy_data = true);

    void destroy_tracked_objects();

    OptixDeviceContext m_context = nullptr;
    OptixDenoiser m_denoiser = nullptr;
    OptixDenoiserParams m_params = {};

    bool m_temporalMode;

    CUdeviceptr m_intensity = 0;
    CUdeviceptr m_avgColor = 0;
    CUdeviceptr m_scratch = 0;
    uint32_t m_scratch_size = 0;
    CUdeviceptr m_state = 0;
    uint32_t m_state_size = 0;

    unsigned int m_tileWidth = 0;
    unsigned int m_tileHeight = 0;
    unsigned int m_overlap = 0;

    OptixDenoiserOptions m_options = {};
    OptixDenoiserModelKind m_modelKind = {};
    OptixDenoiserGuideLayer m_guideLayer = {};

    std::vector<OptixDenoiserLayer> m_layers;
    FilmDenoiserBufferView m_output;

    std::vector<CUdeviceptr> m_trackedCUmemObjects;
};

std::unique_ptr<FilmDenoiser> create_film_optix_denoiser() {
    return std::unique_ptr<FilmDenoiser>{new FilmOptixDenoiser};
}

static void context_log_cb(uint32_t level, const char *tag, const char *message, void * /*cbdata*/) {
    if (level < 4)
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
                  << message << "\n";
}

static OptixPixelFormat filmdenoiser_pixel_format_to_optix_pixel_format(const FilmDenoiserPixelFormat &format, size_t *pixel_size_in_bytes) {

    int pixel_bytes = 0;
    OptixPixelFormat optix_format;
    switch (format) {
        case FILMDENOISER_PIXEL_FORMAT_FLOAT4:
            pixel_bytes = 4 * sizeof(float);
            optix_format = OPTIX_PIXEL_FORMAT_FLOAT4;
            break;
        case FILMDENOISER_PIXEL_FORMAT_UCHAR4:
            pixel_bytes = 4 * sizeof(uint8_t);
            optix_format = OPTIX_PIXEL_FORMAT_UCHAR4;
            break;
        default:
            CUDA_CHECK(("unknown pixel format", cudaErrorInvalidValue));
    }

    if(pixel_size_in_bytes)
        *pixel_size_in_bytes = pixel_bytes;
    return optix_format;
}

FilmOptixDenoiser::~FilmOptixDenoiser() {
    destroy_tracked_objects();
    if (m_denoiser) optixDenoiserDestroy(m_denoiser);
}

void FilmOptixDenoiser::destroy_tracked_objects() {

    for(auto &mem_obj : m_trackedCUmemObjects) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(mem_obj)));
    }

    m_trackedCUmemObjects.clear();
}

// create four channel float OptixImage2D with given dimension. allocate memory on device and
// copy data from host memory given in hmem to device if hmem is nonzero.
OptixImage2D FilmOptixDenoiser::conditional_create_optix_image2d(FilmDenoiserBufferView bv, unsigned int width, unsigned int height, bool copy_data) {
    OptixImage2D oi;
    bool shared_by_ref = false;
    OptixPixelFormat optix_format;
    size_t pixel_size_in_bytes;

    optix_format = filmdenoiser_pixel_format_to_optix_pixel_format(bv.format, &pixel_size_in_bytes);

    const uint64_t frame_byte_size = width * height * pixel_size_in_bytes;

    if(bv.type == FILMDENOISER_BUFFER_VIEW_TYPE_CUDA_DEVICE_MEM) {
        oi.data = static_cast<CUdeviceptr>(bv.address);
        shared_by_ref = true;
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&oi.data), frame_byte_size));
        if (bv.address && copy_data) {
            CUDA_CHECK(bv.type == FILMDENOISER_BUFFER_VIEW_TYPE_HOST_MEM ? cudaSuccess : ("unknown memory type", cudaErrorInvalidValue));
            CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>(oi.data),
                    reinterpret_cast<void *>(bv.address),
                    frame_byte_size,
                    cudaMemcpyHostToDevice));
        }
    }
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * pixel_size_in_bytes;
    oi.pixelStrideInBytes = pixel_size_in_bytes;
    oi.format = optix_format;

    if(!shared_by_ref)
        m_trackedCUmemObjects.push_back(oi.data);
    return oi;
}

int FilmOptixDenoiser::init_device() {

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(nullptr));

        CUcontext cu_ctx = nullptr;// zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &m_context));
    }
    return 0;
}

int FilmOptixDenoiser::init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode) {
    return init_context(data, tileWidth, tileHeight, false, temporalMode);
}

int FilmOptixDenoiser::init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool kpMode, bool temporalMode) {

    m_output = data.output;
    m_temporalMode = temporalMode;

    m_tileWidth = tileWidth > 0 ? tileWidth : data.width;
    m_tileHeight = tileHeight > 0 ? tileHeight : data.height;
    m_temporalMode = temporalMode;

    //
    // cleanup tracked objects
    //
    destroy_tracked_objects();

    //
    // Create denoiser
    //
    {
        OptixDenoiserOptions options = {};
        OptixDenoiserModelKind modelKind = {};

        options.guideAlbedo = data.albedo.address ? 1 : 0;
        options.guideNormal = data.normal.address ? 1 : 0;

        modelKind = temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;

        if (!m_denoiser || m_options.guideAlbedo != options.guideAlbedo || m_options.guideNormal != m_options.guideNormal ||
            m_modelKind != modelKind) {

            if (m_denoiser) optixDenoiserDestroy(m_denoiser);

            OPTIX_CHECK(optixDenoiserCreate(m_context, modelKind, &options, &m_denoiser));
            m_options = options;
            m_modelKind = modelKind;
        }
    }

    //
    // Allocate device memory for denoiser
    //
    {
        OptixDenoiserSizes denoiser_sizes;

        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
                m_denoiser,
                m_tileWidth,
                m_tileHeight,
                &denoiser_sizes));

        if (tileWidth == 0) {
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
            m_overlap = 0;
        } else {
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withOverlapScratchSizeInBytes);
            m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        }

        m_intensity = 0;
        m_avgColor = 0;

        if (kpMode == false) {
            CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void **>(&m_intensity),
                    sizeof(float)));
            m_trackedCUmemObjects.push_back(m_intensity);
        } else {
            CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void **>(&m_avgColor),
                    3 * sizeof(float)));
            m_trackedCUmemObjects.push_back(m_avgColor);
        }

        CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_scratch),
                m_scratch_size));
        m_trackedCUmemObjects.push_back(m_scratch);

        CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_state),
                denoiser_sizes.stateSizeInBytes));
        m_trackedCUmemObjects.push_back(m_state);

        m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

        // clear layers
        m_layers.clear();

        OptixDenoiserLayer layer = {};

        layer.input = conditional_create_optix_image2d(data.color, data.width, data.height);
        layer.output = conditional_create_optix_image2d(data.output, data.width, data.height, false);

        if (m_temporalMode) {
            // this is the first frame, create zero motion vector image
            m_guideLayer.flow = conditional_create_optix_image2d(data.flow, data.width, data.height);

            size_t pixel_size_in_bytes;
            filmdenoiser_pixel_format_to_optix_pixel_format(data.flow.format, &pixel_size_in_bytes);

            // clear flow buffer
            CUDA_CHECK(cudaMemset((void *)m_guideLayer.flow.data, 0, data.width * data.height * pixel_size_in_bytes));

            layer.previousOutput = layer.input;// first frame
        }
        m_layers.push_back(layer);

        if (data.albedo.address)
            m_guideLayer.albedo = conditional_create_optix_image2d(data.albedo, data.width, data.height);
        if (data.normal.address)
            m_guideLayer.normal = conditional_create_optix_image2d(data.normal, data.width, data.height);
    }

    //
    // Setup denoiser
    //
    {
        OPTIX_CHECK(optixDenoiserSetup(
                m_denoiser,
                nullptr,// CUDA stream
                m_tileWidth + 2 * m_overlap,
                m_tileHeight + 2 * m_overlap,
                m_state,
                m_state_size,
                m_scratch,
                m_scratch_size));


        m_params.denoiseAlpha = 0;
        m_params.hdrIntensity = m_intensity;
        m_params.hdrAverageColor = m_avgColor;
        m_params.blendFactor = 0.0f;
    }

    return 0;
}


int FilmOptixDenoiser::update(const FilmDenoiserInputData &data) {

    m_output = data.output;

    auto copy_or_async_mem = [](const FilmDenoiserBufferView &bv, int width, int height, const OptixImage2D &oi) {
        OptixPixelFormat optix_format;
        size_t pixel_size_in_bytes;

        optix_format = filmdenoiser_pixel_format_to_optix_pixel_format(bv.format, &pixel_size_in_bytes);

        CUDA_CHECK(optix_format == oi.format ? cudaSuccess : ("image format is not coincident", cudaErrorInvalidValue));
        CUDA_CHECK(width == oi.width ? cudaSuccess : ("width is not coincident", cudaErrorInvalidValue));
        CUDA_CHECK(height == oi.height ? cudaSuccess : ("height is not coincident", cudaErrorInvalidValue));

        if(bv.format == FILMDENOISER_BUFFER_VIEW_TYPE_CUDA_DEVICE_MEM) {
            if(bv.address == oi.data) {
                // asynchronize it
            } else {
                CUDA_CHECK(cudaMemcpy((void *) oi.data, (void *) bv.address, oi.width * oi.height * pixel_size_in_bytes, cudaMemcpyDeviceToDevice));
            }
        } else {
            CUDA_CHECK(bv.format == FILMDENOISER_BUFFER_VIEW_TYPE_HOST_MEM ? cudaSuccess : ("unknown memory type", cudaErrorInvalidValue));
            CUDA_CHECK(cudaMemcpy((void *) oi.data, (void *) bv.address, oi.width * oi.height * pixel_size_in_bytes, cudaMemcpyHostToDevice));
        }
    };

    copy_or_async_mem(data.color, data.width, data.height, m_layers[0].input);

    if (m_temporalMode) {
        if(data.flow.address)
            copy_or_async_mem(data.flow, data.width, data.height, m_guideLayer.flow);
        m_layers[0].previousOutput = m_layers[0].output;
    }

    if (data.albedo.address)
        copy_or_async_mem(data.albedo, data.width, data.height, m_guideLayer.albedo);

    if (data.normal.address)
        copy_or_async_mem(data.normal, data.width, data.height, m_guideLayer.normal);

    return 0;
}

int FilmOptixDenoiser::exec() {
    if (m_intensity) {
        OPTIX_CHECK(optixDenoiserComputeIntensity(
                m_denoiser,
                nullptr,// CUDA stream
                &m_layers[0].input,
                m_intensity,
                m_scratch,
                m_scratch_size));
    }

    if (m_avgColor) {
        OPTIX_CHECK(optixDenoiserComputeAverageColor(
                m_denoiser,
                nullptr,// CUDA stream
                &m_layers[0].input,
                m_avgColor,
                m_scratch,
                m_scratch_size));
    }

    /**
    OPTIX_CHECK( optixDenoiserInvoke(
                m_denoiser,
                nullptr, // CUDA stream
                &m_params,
                m_state,
                m_state_size,
                &m_guideLayer,
                m_layers.data(),
                static_cast<unsigned int>( m_layers.size() ),
                0, // input offset X
                0, // input offset y
                m_scratch,
                m_scratch_size
                ) );
    **/
    OPTIX_CHECK(optixUtilDenoiserInvokeTiled(
            m_denoiser,
            nullptr,// CUDA stream
            &m_params,
            m_state,
            m_state_size,
            &m_guideLayer,
            m_layers.data(),
            static_cast<unsigned int>(m_layers.size()),
            m_scratch,
            m_scratch_size,
            m_overlap,
            m_tileWidth,
            m_tileHeight));

    CUDA_SYNC_CHECK();

    return 0;
}

int FilmOptixDenoiser::get_results() {

    auto copy_or_async_mem_back = [](const OptixImage2D &oi, const FilmDenoiserBufferView &bv) {
        OptixPixelFormat optix_format;
        size_t pixel_size_in_bytes;

        optix_format = filmdenoiser_pixel_format_to_optix_pixel_format(bv.format, &pixel_size_in_bytes);

        CUDA_CHECK(optix_format == oi.format ? cudaSuccess : ("image format is not coincident", cudaErrorInvalidValue));

        if (bv.format == FILMDENOISER_BUFFER_VIEW_TYPE_CUDA_DEVICE_MEM) {
            if (bv.address == oi.data) {
                // asynchronize it
            } else {
                CUDA_CHECK(cudaMemcpy((void *) bv.address, (void *) oi.data, oi.width * oi.height * pixel_size_in_bytes, cudaMemcpyDeviceToDevice));
            }
        } else {
            CUDA_CHECK(bv.format == FILMDENOISER_BUFFER_VIEW_TYPE_HOST_MEM ? cudaSuccess : ("unknown memory type", cudaErrorInvalidValue));
            CUDA_CHECK(cudaMemcpy((void *) bv.address, (void *) oi.data, oi.width * oi.height * pixel_size_in_bytes, cudaMemcpyDeviceToHost));
        }
    };

    copy_or_async_mem_back(m_layers[0].output, m_output);

    return 0;
}
#endif


};// namespace luminous
