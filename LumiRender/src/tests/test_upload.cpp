//
// Created by Zero on 27/09/2021.
//

#include "core/refl/factory.h"
#include "render/samplers/sampler.h"
#include "core/backend/managed.h"
#include "cpu/cpu_impl.h"

using namespace std;
using namespace luminous;

using lstd::Variant;

class TestSampler : BASE_CLASS(Variant<LCGSampler *, PCGSampler *>) {
public:
    REFL_CLASS(TestSampler)

private:
    using BaseBinder::BaseBinder;
public:
    GEN_BASE_NAME(TestSampler)

    LM_NODISCARD XPU int spp() const {
        LUMINOUS_VAR_PTR_DISPATCH(spp)
    }

    XPU void start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
        LUMINOUS_VAR_PTR_DISPATCH(start_pixel_sample, pixel, sample_index, dimension)
    }

    XPU SensorSample sensor_sample(uint2 p_raster) {
        SensorSample ss;
        ss.p_film = make_float2(p_raster) + next_2d();
        ss.p_lens = next_2d();
        ss.time = next_1d();
        return ss;
    }

    NDSC_XPU float next_1d() {
        LUMINOUS_VAR_PTR_DISPATCH(next_1d)
    }

    NDSC_XPU float2 next_2d() {
        LUMINOUS_VAR_PTR_DISPATCH(next_2d)
    }

    static TestSampler create() {
        return TestSampler(get_arena().create<PCGSampler>(5));
    }
};


REGISTER(TestSampler)

int main() {

    auto device = luminous::create_cpu_device();

    auto &arena = get_arena();

    Managed<TestSampler> sp{device.get()};
    sp.reserve(1);
    auto sampler = TestSampler::create();

//    auto offsets = ClassFactory::instance()->member_offsets(&sampler);

//    sp.push_back(sampler);

    size_t size = 0u;

    arena.for_each_block([&](MemoryArena::ConstBlockIterator block) {
        size += block->usage();
    });

    cout << size << endl;

    auto buffer = device->create_buffer(size);

    cout << arena.description() << endl;

//    for (int i = 0; i < 10; ++i) {
//        cout << sp->next_2d().to_string() << endl;
//    }

    return 0;
}