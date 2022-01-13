#pragma once
#include <cstddef>

namespace luminous {

enum FilmDenoiserBufferViewType {
    FILMDENOISER_BUFFER_VIEW_TYPE_HOST_MEM,
    FILMDENOISER_BUFFER_VIEW_TYPE_CUDA_DEVICE_MEM,
};

enum FilmDenoiserPixelFormat {
    FILMDENOISER_PIXEL_FORMAT_FLOAT4,
    FILMDENOISER_PIXEL_FORMAT_UCHAR4
};

struct FilmDenoiserBufferView {
    FilmDenoiserBufferViewType type;
    FilmDenoiserPixelFormat format;
    unsigned long long address;
};

struct FilmDenoiserInputData {
    uint32_t width = 0;
    uint32_t height = 0;
    FilmDenoiserBufferView color = {};
    FilmDenoiserBufferView albedo = {};
    FilmDenoiserBufferView normal = {};
    FilmDenoiserBufferView flow = {};
    FilmDenoiserBufferView output = {};
    FilmDenoiserBufferView flow_output = {};
};

class FilmDenoiser {
public:
    virtual int init_device() = 0;
    virtual int init_context(const FilmDenoiserInputData &data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode) = 0;
    virtual int update(const FilmDenoiserInputData &data) = 0;
    virtual int exec() = 0;
    virtual int get_results() = 0;
    virtual ~FilmDenoiser() = default;
};

};// namespace luminous
