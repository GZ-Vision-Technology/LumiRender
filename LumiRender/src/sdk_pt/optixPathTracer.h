//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once
#include "gpu/framework/optix_params.h"
#include "gpu/framework/cuda_impl.h"
enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

using Params = luminous::LaunchParams;

struct RayGenData
{
};


struct MissData
{
    luminous::float4 bg_color;
};


struct HitGroupData
{
    luminous::float3  emission_color;
    luminous::float3  diffuse_color;
    luminous::float4* vertices;
};

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

extern "C" char sdk_ptx[];

struct Instance
{
    float transform[12];
};


struct PathTracerState
{
    OptixDeviceContext context = 0;
    luminous::Buffer<OptixInstance> instances{nullptr};
    luminous::Buffer<std::byte> ias_buffer{nullptr};
    luminous::Buffer<std::byte> gas_buffer{nullptr};
    luminous::Buffer<luminous::float3> pos_buffer{nullptr};
    luminous::Buffer<luminous::TriangleHandle> index_buffer{nullptr};

    luminous::Buffer<RayGenRecord> rg_buffer{nullptr};
    luminous::Buffer<MissRecord> ms_rcd_buffer{nullptr};
    luminous::Buffer<HitGroupRecord> hg_rcd_buffer{nullptr};

    std::shared_ptr<luminous::Device> device;
    OptixTraversableHandle         gas_handle               = 0;  // Traversable handle for triangle AS
    OptixTraversableHandle         ias_handle               = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_vertices               = 0;
    CUdeviceptr                    d_tri                    = 0;
    CUdeviceptr                 d_ias_output_buffer          = 0;  // Instance AS memory
    OptixModule                    ptx_module               = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline                 = 0;

    OptixProgramGroup              raygen_prog_group        = 0;
    OptixProgramGroup              radiance_miss_group      = 0;
    OptixProgramGroup              occlusion_miss_group     = 0;
    OptixProgramGroup              radiance_hit_group       = 0;
    OptixProgramGroup              occlusion_hit_group      = 0;

    CUstream                       stream                   = 0;
    Params                         params;
    luminous::Buffer<Params>       b_params{nullptr};
    luminous::Dispatcher dispatcher{std::make_unique<luminous::CUDADispatcher>()};
    OptixShaderBindingTable        sbt                      = {};
};
