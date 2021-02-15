//
// Created by Zero on 2021/2/12.
//


#pragma once

#include "device_memory.h"
#include <memory>

namespace luminous {
    inline namespace gpu {

        struct RangeAllocator {
            int alloc(size_t size);

            void release(size_t begin, size_t size);

            size_t maxAllocedID = 0;
        private:
            struct FreedRange {
                size_t begin;
                size_t size;
            };
            std::vector<FreedRange> freedRanges;
        };

        struct SBT {
            size_t rayGenRecordCount = 0;
            size_t rayGenRecordSize = 0;
            DeviceMemory rayGenRecordsBuffer;

            size_t hitGroupRecordSize = 0;
            size_t hitGroupRecordCount = 0;
            DeviceMemory hitGroupRecordsBuffer;

            size_t missProgRecordSize = 0;
            size_t missProgRecordCount = 0;
            DeviceMemory missProgRecordsBuffer;

            DeviceMemory launchParamsBuffer;
        };

        struct Context;

        /*! optix and cuda context for a single, specific GPU */
        struct DeviceContext : public std::enable_shared_from_this<DeviceContext> {
            typedef std::shared_ptr<DeviceContext> SP;

            /*! create a new device context with given context object, using
                given GPU "cudaID", and serving the rols at the "owlID"th GPU
                in that context */
            DeviceContext(Context *parent,
                          int owlID,
                          int cudaID);

            ~DeviceContext();

            /*! helper function - return cuda name of this device */
            std::string getDeviceName() const;

            /*! helper function - return cuda device ID of this device */
            int getCudaDeviceID() const;

            /*! return the optix default stream for this device. launch params
                may use their own stream */
            CUstream getStream() const { return stream; }

            /*! configures the optixPipeline link options and compile options,
                based on what values (motion blur on/off, multi-level
                instnacing, etc) are set in the context */
            void configurePipelineOptions();

            void buildPrograms();

            void buildMissPrograms();

            void buildRayGenPrograms();

            void buildHitGroupPrograms();

            void destroyPrograms();

            void destroyMissPrograms();

            void destroyRayGenPrograms();

            void destroyHitGroupPrograms();

            void destroyPipeline();

            void buildPipeline();

            /*! collects all compiled programs during 'buildPrograms', such
                that all active progs can then be passed to optix durign
                pipeline creation */
            std::vector<OptixProgramGroup> allActivePrograms;

            OptixDeviceContext optixContext = nullptr;
            CUcontext cudaContext = nullptr;
            CUstream stream = nullptr;

            OptixPipelineCompileOptions pipelineCompileOptions = {};
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            OptixModuleCompileOptions moduleCompileOptions = {};
            OptixPipeline pipeline = nullptr;
            SBT sbt = {};

            /*! the owl context that this device is in */
            Context *const parent;

            /*! linear ID (0,1,2,...) of how *we* number devices (i.e.,
              'first' device is always device 0, no matter if it runs on
              another physical/cuda device) */
            const int ID;

            /* the cuda device ID that this logical device runs on */
            const int cudaDeviceID;
        };

    }
}