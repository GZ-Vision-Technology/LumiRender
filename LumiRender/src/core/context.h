//
// Created by Zero on 2020/9/23.
//

#pragma once

#include <filesystem>
#include <map>
#include <cxxopts.hpp>
#include <core/concepts.h>
#include <core/platform.h>

namespace luminous {

    template<typename T>
    using SP = std::shared_ptr<T>;

    template<typename T>
    using UP = std::unique_ptr<T>;

    class Context : Noncopyable {

    private:
        int _argc;
        const char **_argv;
        std::filesystem::path _run_dir;
        std::filesystem::path _work_dir;
        std::filesystem::path _in_dir;
        std::filesystem::path _scene_file;
        std::string _device;
        int _thread_num{0};
        mutable cxxopts::Options _cli_options;
        mutable std::optional<cxxopts::ParseResult> _parsed_cli_options;
        mutable std::optional<std::string> _positional_option;
        std::map<std::filesystem::path, DynamicModuleHandle, std::less<>> _loaded_modules;

        LM_NODISCARD const cxxopts::ParseResult &_parse_result() const noexcept;

        LM_NODISCARD const std::filesystem::path &_runtime_dir() noexcept;

        LM_NODISCARD const std::filesystem::path &_working_dir() noexcept;

        LM_NODISCARD const std::filesystem::path &_input_dir() noexcept;

        static bool _create_folder_if_necessary(const std::filesystem::path &path) noexcept;

    public:
        Context(int argc, char *argv[]);

        ~Context() noexcept;

        bool has_scene() {
            return _parse_result().count("scene") > 0;
        }

        bool create_working_folder(const std::filesystem::path &name) noexcept;

        bool create_cache_folder(const std::filesystem::path &name) noexcept;

        LM_NODISCARD std::filesystem::path include_path(const std::filesystem::path &name = {}) noexcept;

        LM_NODISCARD std::filesystem::path working_path(const std::filesystem::path &name = {}) noexcept;

        LM_NODISCARD std::filesystem::path runtime_path(const std::filesystem::path &name = {}) noexcept;

        LM_NODISCARD std::filesystem::path input_path(const std::filesystem::path &name = {}) noexcept;

        LM_NODISCARD std::filesystem::path cache_path(const std::filesystem::path &name = {}) noexcept;

        LM_NODISCARD std::filesystem::path scene_path() noexcept;

        LM_NODISCARD const std::filesystem::path &scene_file() noexcept;

        bool has_help_cmd() const noexcept {
            return _parse_result().count("help") > 0;
        }

        LM_NODISCARD bool use_gpu() noexcept {
            return device() == "cuda";
        }

        LM_NODISCARD int thread_num() const noexcept {
            return std::stoi(_parse_result()["thread-num"].as<std::string>());
        }

        void print_help() const noexcept {
            std::cout << _cli_options.help() << std::endl;
        }

        void try_print_help_and_exit()  const noexcept {
            if (has_help_cmd()) {
                print_help();
                exit(0);
            }
        }

        template<typename F>
        LM_NODISCARD auto load_dynamic_function(const std::filesystem::path &path,
                                                 std::string_view module,
                                                 std::string_view function) {
            LUMINOUS_EXCEPTION_IF(module.empty(), "Empty name given for dynamic module");
            auto module_path = std::filesystem::canonical(
                    path / serialize(LUMINOUS_DLL_PREFIX, module, LUMINOUS_DLL_EXTENSION));
            auto iter = _loaded_modules.find(module_path);
            LUMINOUS_INFO(module_path);
            if (iter == _loaded_modules.cend()) {
                iter = _loaded_modules.emplace(module_path, load_dynamic_module(module_path)).first;
            }
            return load_dynamic_symbol<F>(iter->second, std::string{function});
        }

        template<typename T>
        void add_cli_option(const std::string &opt_name, const std::string &desc, const std::string &default_val = {},
                            const std::string &implicit_val = {}) {
            if (!default_val.empty() && !implicit_val.empty()) {
                _cli_options.add_options()(opt_name, desc,
                                           cxxopts::value<T>()->default_value(default_val)->implicit_value(
                                                   implicit_val));
            } else if (!default_val.empty()) {
                _cli_options.add_options()(opt_name, desc, cxxopts::value<T>()->default_value(default_val));
            } else if (!implicit_val.empty()) {
                _cli_options.add_options()(opt_name, desc, cxxopts::value<T>()->implicit_value(implicit_val));
            } else {
                _cli_options.add_options()(opt_name, desc, cxxopts::value<T>());
            }
        }

        template<typename T>
        LM_NODISCARD T cli_option(const std::string &opt_name) const {
            return _parse_result()[opt_name].as<T>();
        }

        LM_NODISCARD std::string cli_positional_option() const {
            return _parse_result()["positional"].as<std::string>();
        }

        LM_NODISCARD std::string device() noexcept;

        LM_NODISCARD bool should_print_generated_source() const noexcept {
            return cli_option<bool>("print-source");
        }
    };

}