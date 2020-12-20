//
// Created by Zero on 2020/9/6.
//

#pragma once

#include <cstdlib>
#include <filesystem>

#include "logging.h"

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))

#include <dlfcn.h>
#include <unistd.h>

#define LUMINOUS_EXPORT [[gnu::visibility("default")]]
#define LUMINOUS_DLL_PREFIX "lib"
#define LUMINOUS_DLL_EXTENSION ".so"

namespace luminous {

inline namespace utility {

    inline size_t memory_page_size() noexcept {
        static thread_local auto page_size = getpagesize();
        return page_size;
    }

    using DynamicModuleHandle = void *;


    inline DynamicModuleHandle load_dynamic_module(const std::filesystem::path &path) {
        LUMINOUS_EXCEPTION_IF_NOT(std::filesystem::exists(path), "Dynamic module not found: ", path);
        LUMINOUS_INFO("Loading dynamic module: ", path);
        auto module = dlopen(std::filesystem::canonical(path).string().c_str(), RTLD_LAZY);
        LUMINOUS_EXCEPTION_IF(module == nullptr, "Failed to load dynamic module ", path, ", reason: ", dlerror());
        return module;
    }

    inline void destroy_dynamic_module(DynamicModuleHandle handle) {
        if (handle != nullptr) {
            dlclose(handle);
        }
    }

    template<typename F>
    inline auto load_dynamic_symbol(DynamicModuleHandle handle, const std::string &name) {
        LUMINOUS_EXCEPTION_IF(name.empty(), "Empty name given for dynamic symbol");
        LUMINOUS_INFO("Loading dynamic symbol: ", name);
        auto symbol = dlsym(handle, name.c_str());
        LUMINOUS_EXCEPTION_IF(symbol == nullptr, "Failed to load dynamic symbol \"", name, "\", reason: ", dlerror());
        return reinterpret_cast<F *>(symbol);
    }
}

}

#elif defined(_WIN32) || defined(_WIN64)

#include <windowsx.h>

#define LUMINOUS_EXPORT __declspec(dllexport)
#define LUMINOUS_DLL_PREFIX ""
#define LUMINOUS_DLL_EXTENSION ".dll"

namespace luminous { inline namespace utility {

using DynamicModuleHandle = HMODULE;

namespace detail {

inline std::string win32_last_error_message() {
    // Retrieve the system error message for the last-error code
    void *buffer = nullptr;
    auto err_code = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&buffer,
        0, nullptr);

    auto err_msg = serialize(buffer, " (code = 0x", std::hex, err_code, ").");
    LocalFree(buffer);

    return err_msg;
}

}// namespace detail

inline size_t memory_page_size() noexcept {
    static thread_local auto page_size = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return info.dwPageSize;
    }();
    return page_size;
}

inline DynamicModuleHandle load_dynamic_module(const std::filesystem::path &path) {
    LUMINOUS_EXCEPTION_IF_NOT(std::filesystem::exists(path), "Dynamic module not found: ", path);
    LUMINOUS_INFO("Loading dynamic module: ", path);
    auto module = LoadLibraryA(std::filesystem::canonical(path).string().c_str());
    LUMINOUS_EXCEPTION_IF(module == nullptr, "Failed to load dynamic module ", path, ", reason: ", detail::win32_last_error_message());
    return module;
}

inline void destroy_dynamic_module(DynamicModuleHandle handle) {
    if (handle != nullptr) { FreeLibrary(handle); }
}

template<typename F>
inline auto load_dynamic_symbol(DynamicModuleHandle handle, const std::string &name) {
    LUMINOUS_EXCEPTION_IF(name.empty(), "Empty name given for dynamic symbol");
    LUMINOUS_INFO("Loading dynamic symbol: ", name);
    auto symbol = GetProcAddress(handle, name.c_str());
    LUMINOUS_EXCEPTION_IF(symbol == nullptr, "Failed to load dynamic symbol \"", name, "\", reason: ", detail::win32_last_error_message());
    return reinterpret_cast<F *>(symbol);
}

}}// namespace luminous::utility

#else
#error Unsupported platform for DLL exporting and importing
#endif