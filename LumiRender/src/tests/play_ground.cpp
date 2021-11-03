#include "util/parallel.h"
#include "gpu/framework/jitify/jitify.hpp"
#include "iostream"
#include "core/context.h"

using namespace luminous;

using namespace std;



int main(int argc, char *argv[]) {

    Context context{argc, argv};

    string fn = "E:\\work\\graphic\\renderer\\LumiRender\\LumiRender\\src\\gpu\\shaders\\megakernel_pt.cu";

    auto content = context.load_cu_file(fn);



    cout << content << endl;

}