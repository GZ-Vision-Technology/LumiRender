# Build via CMake

We use CMake-3.19+ and `CMakePresets.json` to config cmake generator system and build environt. Althrough
all the settings in `CMakePresets.json` is explict referring [cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html#build-preset), but some implict system-wide variable must be set before
you can build the whole project. We will cover those details by each of supported (development+runtime) platform(NOTE: _cross-platform building have not been tested, so we will not cover it here_).

## Windows
 * CMake: installation version must be at least v3.19, v3.22 is tested that will work, Add CMake bin path to your `PATH` variable;
 * Visual C++ development environment: intall Visual Studio 2019 IDE or obove with C++ workload selecting
  [Desktop development with C++](https://docs.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-community?view=vs-2019&preserve-view=true) will be far more than enough;
 * Vcpkg: install via [vcpkg.git](https://github.com/Microsoft/vcpkg), than check the following variables are set to correct locations:
    + `VCPKG_ROOT` set to path of Vcpkg local repo;
    + `PATH` concatenate `VCPKG_ROOT` to `PATH` variable (Optional);
 * Install Cuda-11.4 SDK and OptiX SDK, ensure the following variables are set correctly:
    + `CUDA_PATH` set to `<install prefix>\NVIDIA GPU Computing Toolkit\CUDA\v11.4`;
    + `CUDA_PATH_V11_4` set to `<install prefix>\NVIDIA GPU Computing Toolkit\CUDA\v11.4`;
    + `PATH` concatenate `C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.3.0`;

## Linux-x86_64
 ### Prerequistes
   * [GNU GCC compiler 9.3.0+](https://gcc.gnu.org/install/), If you using Ubuntu distro 16.04 LTS(Xenial) and later, we recomand[GCC 9 - defaults (Xenial & Bionic)](https://launchpad.net/~savoury1/+archive/ubuntu/gcc-defaults-9);
   * [Clang-11+ front compiler](https://github.com/llvm/llvm-project/releases/tag/llvmorg-11.1.0);
   * [CUDA SDK 11.3.0+](https://developer.nvidia.com/cuda-downloads), Be carefull to choose the available SDK version compatible with you Linux distro version;
   * [OptiX SDK 7.3.0+](https://developer.nvidia.com/optix);
   * [CMake 3.22+](https://cmake.org/download/);
   * [Ninja-build 1.10.2+](https://ninja-build.org/);

 ### Install Third Party Libraries
   * [embree3](https://github.com/embree/embree/releases);
   * [Assimp 5.0.1](http://assimp.org/index.php/downloads);

 ### Set Environment Variables
you must set the following variables:
 ```bash
# ninja-build
NINJA_ROOT=/home/wangyonghong/build-tools/ninja

VCPKG_ROOT=/home/wangyonghong/build-tools/vcpkg
export VCPKG_ROOT

# Assimp
ASSIMP_INSTALL_DIR=/home/wangyonghong/installed/assimp-5.0.1/installed/
export ASSIMP_INSTALL_DIR

# Cuda
CUDA_PATH="/usr/local/cuda-11.3"
CUDA_PATH_V11_3=$CUDA_PATH
export CUDA_PATH
export CUDA_PATH_V11_3


# OptiX
OptiX_INSTALL_DIR=/home/wangyonghong/nv/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
export OptiX_INSTALL_DIR


# embree3
EMBREE3_INSTALL_DIR="/home/wangyonghong/installed/embree-3.13.1.x86_64.linux"
export EMBREE3_INSTALL_DIR

# CPATH="${EMBREE3_INSTALL_DIR}/include:${CPATH}"
# export CPATH

# LIBRARY_PATH="${EMBREE3_INSTALL_DIR}/lib:${LIBRARY_PATH}"
# export LIBRARY_PATH

# LD_LIBRARY_PATH="${EMBREE3_INSTALL_DIR}/lib:/lib:/usr/lib:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH


# Runtime variables

PATH=$NINJA_ROOT:$VCPKG_ROOT:/usr/local/cuda-11.3/bin:$PATH
export PATH

LD_LIBRARY_PATH="$ASSIMP_INSTALL_DIR/installed/Debug/lib:$ASSIMP_INSTALL_DIR/installed/Release/lib:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
 ```

There are many approach to do it, we recommand:
 * concatenate above _bash_ content into `~/.profile`:
  ```bash
  vi ~/.profile
  # <Copy above bash code and save the file>
  source ~/.profile
  ```
 * If you are super user(root), you can concatenate above bash code into `/etc/environment`. By this way, all the users use those variables.

 ### Build
 We use [cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to simplify the configuration procedure, to list available configuration presets in `CMakePresets.json`, change your work directory to source directory and execute:
 ```bash
 cmake --list-presets
 ```
 If you want to use one of available preset: `<preset-name>`, config it using:
 ```
 cmake --preset=<preset-name>
 ```
 To build the generated project:
 ```
 cmake --build <path to generate project directory>
 ```

 ### Runtime dependencies
 ```
ln -sf /lib/x86_64-linux-gnu/librt.so.1 librt.so.1
ln -sf /lib/x86_64-linux-gnu/libdl.so.2 libdl.so.2
ln -sf /usr/lib/x86_64-linux-gnu/libX11.so.6 libX11.so.6
ln -sf /lib/x86_64-linux-gnu/libpthread.so.0 libpthread.so.0
ln -sf /lib/x86_64-linux-gnu/libm.so.6 libm.so.6
ln -sf /lib/x86_64-linux-gnu/libgcc_s.so.1 libgcc_s.so.1
ln -sf /lib/x86_64-linux-gnu/libc.so.6 libc.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libxcb.so.1 libxcb.so.1
ln -sf /usr/lib/x86_64-linux-gnu/libXau.so.6 libXau.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 libXdmcp.so.6

```