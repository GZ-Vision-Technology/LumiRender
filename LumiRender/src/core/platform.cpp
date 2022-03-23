#include "platform.h"
#include <array>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include <cstring>
#include <vector>

namespace luminous {

InstructionSet::InstructionSet_Internal const& InstructionSet::CPU_Rep() {
    static InstructionSet_Internal CPU_RepData;
    return CPU_RepData;
}

InstructionSet::InstructionSet_Internal::InstructionSet_Internal()
    : nIds_{0},
      nExIds_{0},
      isIntel_{false},
      isAMD_{false},
      f_1_ECX_{0},
      f_1_EDX_{0},
      f_7_EBX_{0},
      f_7_ECX_{0},
      f_81_ECX_{0},
      f_81_EDX_{0}
{

    std::vector<std::array<int, 4>> data;
    std::vector<std::array<int, 4>> extdata;

    //int cpuInfo[4] = {-1};
    std::array<int, 4> cpui;

    // Calling __cpuid with 0x0 as the function_id argument
    // gets the number of the highest valid function ID.
#ifdef _MSC_VER
    __cpuid(cpui.data(), 0);
    nIds_ = cpui[0];
#else
    nIds_ = __get_cpuid_max(0, nullptr);
#endif

    for (int i = 0; i <= nIds_; ++i) {
#ifdef _MSC_VER
        __cpuidex(cpui.data(), i, 0);
#else
        __get_cpuid(i, (unsigned int*)&cpui[0], (unsigned int*)&cpui[1], (unsigned int*)&cpui[2], (unsigned int*)&cpui[3]);
#endif
        data.push_back(cpui);
    }

    // Capture vendor string
    char vendor[0x20];
    memset(vendor, 0, sizeof(vendor));
    *reinterpret_cast<int *>(vendor) = data[0][1];
    *reinterpret_cast<int *>(vendor + 4) = data[0][3];
    *reinterpret_cast<int *>(vendor + 8) = data[0][2];
    vendor_ = vendor;
    if (vendor_ == "GenuineIntel") {
        isIntel_ = true;
    } else if (vendor_ == "AuthenticAMD") {
        isAMD_ = true;
    }

    // load bitset with flags for function 0x00000001
    if (nIds_ >= 1) {
        f_1_ECX_ = data[1][2];
        f_1_EDX_ = data[1][3];
    }

    // load bitset with flags for function 0x00000007
    if (nIds_ >= 7) {
        f_7_EBX_ = data[7][1];
        f_7_ECX_ = data[7][2];
    }

    // Calling __cpuid with 0x80000000 as the function_id argument
    // gets the number of the highest valid extended ID.
#if _MSC_VER
    __cpuid(cpui.data(), 0x80000000);
    nExIds_ = cpui[0];
#else
    nExIds_ = __get_cpuid_max(0x80000000, (unsigned int*)&cpui[0]);
#endif

      char brand[0x40];
      memset(brand, 0, sizeof(brand));

      for (int i = 0x80000000; i <= nExIds_; ++i) {
#if _MSC_VER
          __cpuidex(cpui.data(), i, 0);
#else
          __get_cpuid(i, (unsigned int*)&cpui[0], (unsigned int*)&cpui[1], (unsigned int*)&cpui[2], (unsigned int*)&cpui[3]);
#endif
          extdata.push_back(cpui);
      }

      // load bitset with flags for function 0x80000001
      if (nExIds_ >= 0x80000001) {
          f_81_ECX_ = extdata[1][2];
          f_81_EDX_ = extdata[1][3];
      }

      // Interpret CPU brand string if reported
      if (nExIds_ >= 0x80000004) {
          memcpy(brand, extdata[2].data(), sizeof(cpui));
          memcpy(brand + 16, extdata[3].data(), sizeof(cpui));
          memcpy(brand + 32, extdata[4].data(), sizeof(cpui));
          brand_ = brand;
      }
}

}// namespace luminous

namespace luminous {

    luminous_fs::path get_current_executable_directory() {

        std::string path_buffer;

#if defined(_WIN32)
        DWORD rc;

        path_buffer.resize(MAX_PATH);

        do {
            rc = GetModuleFileNameA(NULL, path_buffer.data(), path_buffer.length());
            if(rc == 0)
                throw std::system_error(std::error_code(
                    GetLastError(), std::system_category()
                ), "[Win32]Get current executable path failed");
            else if(rc < path_buffer.length())
                break;

            path_buffer.append(80, '\0');
            rc = GetLastError();

        } while (rc == ERROR_INSUFFICIENT_BUFFER);
#elif defined(__linux__)
        int rc;

        path_buffer.reize(MAX_PATH);

        do {
            rc = readlink("proc/self/exe", path_buffer.data(), path_buffer.length());

            if(rc < 0)
                throw std::system_error(std::error_code(
                                                errno, std::system_category()),
                                        "[Lunix]Get current executable path failed");
            else if (rc < path_buffer.length())
                break;

            path_buffer.append(80, '\0');
            rc = 0;
        } while (!rc);
#else
        static_assert(0, "Unsupported platform!");
#endif

        luminous_fs::path exe_path(std::move(path_buffer));
        return exe_path.parent_path();
    }

}// namespace luminous
