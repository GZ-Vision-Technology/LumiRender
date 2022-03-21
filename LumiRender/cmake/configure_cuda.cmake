## ======================================================================== ##
## Copyright 2018-2019 Ingo Wald                                            ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# Verbosely echo NVCC command line
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
find_package(CUDA REQUIRED)

# OptiX debugging is not supported yet.
# so --device-debug or -G option should not be here.
# @see https://forums.developer.nvidia.com/t/compile-error-in-debug/158255/3
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

if (CUDA_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()
