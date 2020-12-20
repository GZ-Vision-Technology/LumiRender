//
// Created by Zero on 2020/9/1.
//

#pragma once


#include <string>
#include <string_view>
#include <sstream>
#include <fstream>
#include <array>
#include <regex>
#include <filesystem>

namespace luminous {
    inline namespace utility {

        template<typename ...Args>
        inline std::string serialize(Args &&...args) noexcept {
            std::ostringstream ss;
            static_cast<void>((ss << ... << std::forward<Args>(args)));
            return ss.str();
        }

        inline std::string pascal_to_snake_case(std::string_view s) noexcept {  // TODO: Robustness
            std::string result;
            auto lower_met = false;
            for (auto i = 0u; i < s.size(); i++) {
                auto c = s[i];
                if (std::isupper(c)) {
                    if (lower_met || (i != 0u && i != s.size() - 1u && !std::isupper(s[i + 1]))) { result.push_back('_'); }
                    lower_met = false;
                } else {
                    lower_met = true;
                }
                result.push_back(std::tolower(c));
            }
            return result;
        }

        inline std::string text_file_contents(const std::filesystem::path &file_path) {
            std::ifstream file{file_path};
            if (!file.is_open()) {
                std::ostringstream ss;
                ss << "Failed to open file: " << file_path;
                throw std::runtime_error{ss.str()};
            }
            return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
        }

        inline std::string to_lower(std::string s) noexcept {
            for (auto &&c : s) { c = static_cast<char>(std::tolower(c)); }
            return s;
        }

        inline std::string jsonc_to_json(const std::string &jsonc) {
            std::string json = std::regex_replace(jsonc, std::regex(R"(\/\/.*\n)"), "\n");
            return json;
        }

        inline void string_printf_recursive(std::string *s, const char *fmt) {
            const char *c = fmt;
            // No args left; make sure there aren't any extra formatting
            // specifiers.
            while (*c) {
                if (*c == '%') {
                    ++c;
                }
                *s += *c++;
            }
        }

        // 1. Copy from fmt to *s, up to the next formatting directive.
        // 2. Advance fmt past the next formatting directive and return the
        //    formatting directive as a string.
        inline std::string copy_to_format_string(const char **fmt_ptr, std::string *s) {
            const char *&fmt = *fmt_ptr;
            while (*fmt) {
                if (*fmt != '%') {
                    *s += *fmt;
                    ++fmt;
                } else if (fmt[1] == '%') {
                    // "%%"; let it pass through
                    *s += '%';
                    *s += '%';
                    fmt += 2;
                } else
                    // fmt is at the start of a formatting directive.
                    break;
            }

            std::string nextFmt;
            if (*fmt) {
                do {
                    nextFmt += *fmt;
                    ++fmt;
                    // Incomplete (but good enough?) test for the end of the
                    // formatting directive: a new formatting directive starts, we
                    // hit whitespace, or we hit a comma.
                } while (*fmt && *fmt != '%' && !isspace(*fmt) && *fmt != ',' &&
                         *fmt != '[' && *fmt != ']' && *fmt != '(' && *fmt != ')');
            }

            return nextFmt;
        }

        template <typename T>
        inline std::string format_one(const char *fmt, T v) {
            // Figure out how much space we need to allocate; add an extra
            // character for the '\0'.
            size_t size = snprintf(nullptr, 0, fmt, v) + 1;
            std::string str;
            str.resize(size);
            snprintf(&str[0], size, fmt, v);
            str.pop_back();  // remove trailing NUL
            return str;
        }

        // General-purpose version of string_printf_recursive; add the formatted
        // output for a single string_printf() argument to the final result string
        // in *s.
        template <typename T, typename... Args>
        inline void string_printf_recursive(std::string *s, const char *fmt, T v,
                                          Args... args) {
            std::string nextFmt = copy_to_format_string(&fmt, s);
            *s += format_one(nextFmt.c_str(), v);
            string_printf_recursive(s, fmt, args...);
        }

        // Special case of StringPrintRecursive for float-valued arguments.
        template <typename... Args>
        inline void string_printf_recursive(std::string *s, const char *fmt, float v,
                                          Args... args) {
            std::string nextFmt = copy_to_format_string(&fmt, s);
            if (nextFmt == "%f")
                // Always use enough precision so that the printed value gives
                // the exact floating-point value if it's used to initialize a
                // float.
                // https://randomascii.wordpress.com/2012/03/08/float-precisionfrom-zero-to-100-digits-2/
                *s += format_one("%.9g", v);
            else
                // If a specific formatting string other than "%f" was specified,
                // just use that.
                *s += format_one(nextFmt.c_str(), v);

            // Go forth and print the next arg.
            string_printf_recursive(s, fmt, args...);
        }

        // Specialization for doubles that always uses enough precision.  (It seems
        // that this is the version that is actually called for floats.  I thought
        // that float->double promotion wasn't supposed to happen in this case?)
        template <typename... Args>
        inline void string_printf_recursive(std::string *s, const char *fmt, double v,
                                          Args... args) {
            std::string nextFmt = copy_to_format_string(&fmt, s);
            if (nextFmt == "%f")
                *s += format_one("%.17g", v);
            else
                *s += format_one(nextFmt.c_str(), v);
            string_printf_recursive(s, fmt, args...);
        }

        // string_printf() is a replacement for sprintf() (and the like) that
        // returns the result as a std::string. This gives convenience/control
        // of printf-style formatting in a more C++-ish way.
        //
        // Floating-point values with the formatting string "%f" are handled
        // specially so that enough digits are always printed so that the original
        // float/double can be reconstituted exactly from the printed digits.
        template <typename... Args>
        inline std::string string_printf(const char *fmt, Args... args) {
            std::string ret;
            string_printf_recursive(&ret, fmt, args...);
            return ret;
        }
    }
}

#ifdef __OBJC__

#import <Foundation/Foundation.h>

namespace luminous { inline namespace utility {

[[nodiscard]] inline NSString *make_objc_string(const char *s) noexcept {
    return [[NSString alloc] initWithCString:s encoding:NSUTF8StringEncoding];
}

[[nodiscard]] inline NSString *make_objc_string(std::string_view sv) noexcept {
    return [[NSString alloc] initWithBytes:sv.data() length:sv.size() encoding:NSUTF8StringEncoding];
}

[[nodiscard]] inline std::string to_string(NSString *s) noexcept {
    return s.UTF8String;
}

[[nodiscard]] inline std::string_view to_string_view(NSString *s) noexcept {
    return {s.UTF8String, s.length};
}

}}

#endif


