include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/stblib/stb")
include_directories("${PROJECT_SOURCE_DIR}/src/ext")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui")
include_directories("${PROJECT_SOURCE_DIR}/src/ext/gui/imgui/glad")

include_directories("${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cxxopts/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/assimp/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/embree/include")

include_directories("${CMAKE_BINARY_DIR}/thirdparty/assimp/include")

include_directories("${LUMINOUS_VCPKG_PATH}/include")

if(LUMINOUS_LOCAL_EMBREE)

else()
    include_directories("${CMAKE_BINARY_DIR}/thirdparty/embree/include")
endif()

if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_library(luminous-all INTERFACE)
    if (LUMINOUS_CPU_BACKEND)
        target_link_libraries(luminous-all INTERFACE
                luminous-core
                luminous-ext
                luminous-gpu
                luminous-cpu
                luminous-render
                luminous-base_libs
                luminous-view
                luminous-util)
    else()
        target_link_libraries(luminous-all INTERFACE
                luminous-core
                luminous-ext
                luminous-gpu
                luminous-render
                luminous-base_libs
                luminous-view
                luminous-util)
    endif ()

else()
    add_library(luminous-all INTERFACE)
    target_link_libraries(luminous-all INTERFACE
            luminous-core
            luminous-ext
            luminous-cpu
            luminous-render
            luminous-base_libs
            luminous-view
            luminous-util)
endif ()



add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(ext)
if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    add_definitions (-D CUDA_SUPPORT)
    add_subdirectory(gpu)
endif ()
if (LUMINOUS_CPU_BACKEND)
    add_subdirectory(cpu)
endif ()
add_subdirectory(view)
add_subdirectory(base_libs)
add_subdirectory(tests)
add_subdirectory(apps)
add_subdirectory(render)

#if(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)


#endif()