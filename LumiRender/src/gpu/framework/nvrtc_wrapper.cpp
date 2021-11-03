//
// Created by Zero on 03/11/2021.
//

#include "nvrtc_wrapper.h"
#include "core/context.h"

namespace luminous {
    inline namespace gpu {

        void NvrtcWrapper::add_included_path(const string &path) {
            _included_path.push_back(path);
        }

        string NvrtcWrapper::compile_cu_src_to_ptx(const string &cu_src) {
            // Create program
            nvrtcProgram prog = 0;

            CHECK_NVRTC(nvrtcCreateProgram(&prog, cu_src.c_str(), "name", 0, nullptr, nullptr));

            CHECK_NVRTC(nvrtcCompileProgram(prog, (int) _compile_options.size(), _compile_options.data()));

            size_t log_size = 0;
            CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
            string log;
            log.resize(log_size);

            return std::string();
        }

        string NvrtcWrapper::get_ptx_from_cu_file(luminous_fs::path fn) {
            string cu_src = _context->load_cu_file(std::move(fn));

            return compile_cu_src_to_ptx(cu_src);
        }
    }
}