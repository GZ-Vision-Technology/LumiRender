#include "util/parallel.h"
#include "iostream"
#include "core/context.h"
#include "gpu/framework/nvrtc_wrapper.h"

using namespace luminous;

using namespace std;



int main(int argc, char *argv[]) {

    Context context{argc, argv};

    string fn = "E:\\work\\graphic\\renderer\\LumiRender\\LumiRender\\src\\gpu\\shaders\\test_kernels.cu";

    string path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda\\std\\detail\\libcxx\\include";

//    path = "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.18362.0\\ucrt";

    NvrtcWrapper nvrtc_wrapper{&context};

    nvrtc_wrapper.add_included_path(path);

    nvrtc_wrapper.compile_cu_file_to_ptx(fn);


}