//
// Created by Zero on 03/11/2021.
//

#include "nvrtc_wrapper.h"
#include "core/context.h"

#define CUDA_NVRTC_OPTIONS  \
"-std=c++17", \
"-arch", \
"compute_60", \
"-use_fast_math", \
"-lineinfo", \
"-default-device", \
"-rdc", \
"true", \
"-D__x86_64",

namespace luminous {
    inline namespace gpu {

        void NvrtcWrapper::add_included_path(const string &path) {
            _included_path.push_back(path);
        }

        string NvrtcWrapper::compile_cu_file_to_ptx(const string &fn) {
            // Create program
            nvrtcProgram prog = nullptr;
            string cu_src = _context->load_cu_file(std::move(fn));

            CHECK_NVRTC(nvrtcCreateProgram(&prog, cu_src.c_str(), fn.c_str(), 0, nullptr, nullptr));

            auto fnl = std::string( "-I" ) + _included_path[0];

            _compile_options.push_back(fnl.c_str());

            nvrtcCompileProgram(prog, (int) _compile_options.size(), _compile_options.data());

            size_t log_size = 0;
            CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
            if (log_size > 0) {
                string log;
                log.resize(log_size);
                CHECK_NVRTC(nvrtcGetProgramLog(prog, &log[0]));
                std::cout << log << std::endl;;
            }

            size_t ptx_size = 0;
            CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
            string ptx;
            ptx.resize(ptx_size);
            CHECK_NVRTC(nvrtcGetPTX(prog, &ptx[0]));


            return ptx;
        }
    }
}