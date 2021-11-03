//
// Created by Zero on 03/11/2021.
//


#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>  // For dim3, cudaStream_t
#include <map>
#include <string>
#include "core/platform.h"

#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

#define CHECK_NVRTC(call)                              \
    do {                                               \
        nvrtcResult code = call;      \
        if (code != NVRTC_SUCCESS) {  \
        throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) );         \
        }                                              \
    } while (0);

namespace luminous {

    class Context;

    inline namespace gpu {
        using std::string;

        using std::map;

        class NvrtcWrapper {
        private:
            Context * _context{nullptr};
            map<string, string> _cu_cache;
            std::vector<const char *> _compile_options;
            std::vector<string> _included_path;
        public:

            explicit NvrtcWrapper(Context *context) : _context(context) {}

            NvrtcWrapper() = default;

            void add_included_path(const string &path);

            string compile_cu_src_to_ptx(const string &cu_src);

            string get_ptx_from_cu_file(luminous_fs::path fn);
        };
    }
}