cmake_minimum_required(VERSION 3.17)

project(LumiRender LANGUAGES C CXX ISPC VERSION 1.0.0)

# Option for building test targets
option(LUMINOUS_BUILD_TESTS "Build tests" FALSE)

if(UNIX)
  add_compile_options("-fPIC")
  # don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  # when building, don't use the install RPATH already available
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  # the RPATH to be used when installing
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

  # don't add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

# VLD
option( LUMINOUS_ENABLE_VLD "Enable VLD for debug build" FALSE)
if (WIN32 AND ${LUMINOUS_ENABLE_VLD})
    if(NOT DEFINED ENV{VLD_SDK_PATH})
        message(FATAL_ERROR "Can not find VLD SDK $ENV{VLD_SDK_PATH} in build environment!")
    else()
        message(STATUS "VLD SDK found in '$ENV{VLD_SDK_PATH}'")
    endif()

    include_directories($ENV{VLD_SDK_PATH}/include)
    link_directories($ENV{VLD_SDK_PATH}/lib/Win64)
endif(WIN32 AND ${LUMINOUS_ENABLE_VLD})

set(STDCXX_EXTRA_LIBS "")

# include(CheckCXXSymbolExists)
# check_cxx_symbol_exists(std::filesystem::path::preferred_separator filesystem cxx17fs)
# if(NOT cxx17fs)
#     message("std::filesystem will be replaced by std::experimental::filesystem in gcc")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least gcc 7.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
        # add_link_options("-lstdc++fs" "-lpthread")
        list(APPEND STDCXX_EXTRA_LIBS "stdc++fs" "pthread")
    endif()
# endif(NOT cxx17fs)
endif()


# 项目信息
set(APP_NAME LumiRender CACHE STRING INTERNAL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    include(configure_optix)
    include(CheckLanguage)
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT ON)
else ()
    set(LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT OFF)
    message(not find cuda !!)
endif ()

set(CMAKE_CXX_STANDARD 11)

function(luminous_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luminous-all ${STDCXX_EXTRA_LIBS})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/")
endfunction()

find_package(assimp CONFIG REQUIRED)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp)

set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)

set(CMAKE_DEBUG_POSTFIX "")

set(LUMINOUS_CPU_BACKEND ON)
set(LUMINOUS_LOCAL_EMBREE ON)

if (LUMINOUS_LOCAL_EMBREE)
    find_package(embree 3 CONFIG REQUIRED)
else()
#    add_subdirectory(thirdparty/embree)
#    add_library(embree_config INTERFACE)
#    target_link_libraries(embree_config INTERFACE embree3)
#    set_target_properties(embree PROPERTIES
#            PUBLIC_HEADER "${HEADER_FILES}"
#            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Disable OpenImageIO for Unix like build(for imcompatible boost version conflicts)
if(NOT UNIX)
    find_package(OpenImageIO CONFIG REQUIRED)
endif()

find_package(OpenImageDenoise CONFIG REQUIRED)

if ("$<CONFIG>" MATCHES Debug)
    add_definitions (-D DEBUG_BUILD)
endif ()

set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
add_subdirectory(src)

set (CHECK_CUDA_OUTPUT_EXE ${CMAKE_BINARY_DIR}/check_cuda)
if (MSVC)
  if(NOT EXISTS "${CHECK_CUDA_OUTPUT_EXE}.exe")
    execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/check_cuda.cu -ccbin ${CMAKE_CXX_COMPILER} -o ${CHECK_CUDA_OUTPUT_EXE})
  endif()
else  ()
  if(NOT EXISTS "${CHECK_CUDA_OUTPUT_EXE}")
    execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/check_cuda.cu -o ${CHECK_CUDA_OUTPUT_EXE})
  endif()
endif ()

execute_process (COMMAND ${CHECK_CUDA_OUTPUT_EXE}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE CHECK_CUDA_OUTPUT)

if (LUMINOUS_RENDER_ENABLE_CUDA_DEFAULT)
    message(${CHECK_CUDA_OUTPUT_EXE})
    message(${CUDA_RETURN_CODE})
    message(${CHECK_CUDA_OUTPUT})

    if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
        message (SEND_ERROR ${CHECK_CUDA_OUTPUT})
    else ()
        set(ARCH "${CHECK_CUDA_OUTPUT}")
        message (STATUS "Detected CUDA Architecture: ${ARCH}")
        string (APPEND CMAKE_CUDA_FLAGS " --gpu-architecture=${ARCH}")
    endif ()
endif ()

# --Package
include(CMakePackageConfigHelpers)

string(TOLOWER ${PROJECT_NAME} PACKAGE_NAME)

set(LUMIRENDERER_MULTIPLE_TARGETS
  luminous
  luminous-core
  luminous-cpu
  luminous-ext-stb
  luminous-ext-tinyexr
  luminous-gpu
  luminous-render
  luminous-parser
  luminous-util
  luminous-denoise
  luminous-base_libs
  luminous-ext
)

install(
  TARGETS ${LUMIRENDERER_MULTIPLE_TARGETS}
  RUNTIME_DEPENDENCY_SET LUMINOUS_RUNTIME_DEPS
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(WIN32)
  set(LUMINOUS_RUNTIME_DEPS_EXCLUDES_PREFIX "api-ms-" "ext-ms-")
  set(LUMINOUS_RUNTIME_DEPS_EXCLUDES_POSTFIX ".*system32/.*\\.dll" "cuda")
else()
  set(LUMINOUS_RUNTIME_DEPS_EXCLUDES_PREFIX "")
  set(LUMINOUS_RUNTIME_DEPS_EXCLUDES_POSTFIX "librt.*" "libdl.*" "libX11.*" "libpthread.*" "libm.*" "libgcc_s.*" "libc\..*"
    "libxcb.*" "libXau.*" "libXdmcp.*" "libz.*" "ld-.*" "libcuda\..*")
endif()

install(RUNTIME_DEPENDENCY_SET LUMINOUS_RUNTIME_DEPS
  PRE_EXCLUDE_REGEXES  ${LUMINOUS_RUNTIME_DEPS_EXCLUDES_PREFIX}
  POST_EXCLUDE_REGEXES ${LUMINOUS_RUNTIME_DEPS_EXCLUDES_POSTFIX}
  DIRECTORIES ${DEPENDENCY_PATHS}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# install(CODE "set(DEPENDENCY_PATHS \"${DEPENDENCY_PATHS}\")")
# install(
#   CODE [[
#     file(GET_RUNTIME_DEPENDENCIES
#       EXECUTABLES $<TARGET_FILE:luminous>
#       RESOLVED_DEPENDENCIES_VAR _r_deps
#       UNRESOLVED_DEPENDENCIES_VAR _u_deps
#       DIRECTORIES ${DEPENDENCY_PATHS}
#       PRE_EXCLUDE_REGEXES  "api-ms-" "ext-ms-"
#       POST_EXCLUDE_REGEXES  ".*system32/.*\\.dll"
#     )
#     foreach(_file ${_r_deps})
#         file(INSTALL
#           DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
#           TYPE SHARED_LIBRARY
#           FOLLOW_SYMLINK_CHAIN
#           FILES "${_file}"
#         )
#     endforeach()
#     list(LENGTH _u_deps _u_length)
#     if("${_u_length}" GREATER 0)
#       message(WARNING "Unresolved dependencies detected!")
#       foreach(_file ${_u_deps})
#         message(WARNING "Unresolved dependency: \"${_file}\"")
#       endforeach()
#     endif()
#   ]])

# Auto package
include(CPack)






