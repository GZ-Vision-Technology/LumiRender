
#file(GLOB SOURCE_FILES *.h *.c*)


file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(luminous-gpu SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(luminous-gpu
        PUBLIC CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::cudart
        luminous-graphics
        luminous-core
        luminous-render
        )

set_target_properties(luminous-gpu PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)




#luminous_render_add_backend(cuda SOURCES ${SOURCE_FILES})
#target_include_directories(luminous-backend-cuda PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

#target_link_libraries(luminous-backend-cuda
#        PRIVATE CUDA::nvrtc
#        CUDA::cuda_driver
#        CUDA::cudart
#        luminous-graphics
#        )

#set_target_properties(luminous-backend-cuda PROPERTIES
#        PUBLIC_HEADER "${HEADER_FILES}"
#        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#        WINDOWS_EXPORT_ALL_SYMBOLS ON)