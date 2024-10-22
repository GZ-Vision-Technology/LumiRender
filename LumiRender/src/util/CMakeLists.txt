
set(HEADER_FILES
  image_base.h
  image.h
  parallel.h
  progressreporter.h
  stats.h
)

set(SOURCE_FILES
  image.cpp
  parallel.cpp
  progressreporter.cpp
  stats.cpp
)

set(ISPC_SOURCE_FILES
  image_util.ispc   
)

add_library(luminous-util SHARED ${HEADER_FILES} ${SOURCE_FILES}
${ISPC_SOURCE_FILES}
)

if(WIN32)
target_link_libraries(luminous-util
        PUBLIC
        luminous-core
        luminous-parser
        PRIVATE
        CUDA::cudart
        Ws2_32.lib
        )
endif()

if(UNIX)
target_link_libraries(luminous-util PUBLIC
        luminous-core
        luminous-parser
        PRIVATE
        CUDA::cudart
        )
endif()

set_target_properties(luminous-util PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
