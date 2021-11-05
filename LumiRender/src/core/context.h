//
// Created by Zero on 2020/9/23.
//

#pragma once

#include <filesystem>
#include <map>
#include <cxxopts.hpp>
#include <core/concepts.h>
#include <core/platform.h>

namespace fs = std::filesystem;

namespace luminous {

    template<typename T>
    using SP = std::shared_ptr<T>;

    template<typename T>
    using UP = std::unique_ptr<T>;

    class Context : Noncopyable {

    private:
        int _argc;
        const char **_argv;
        luminous_fs::path _run_dir;
        luminous_fs::path _work_dir;
        luminous_fs::path _in_dir;
        luminous_fs::path _scene_file;
        std::string _device;
        mutable cxxopts::Options _cli_options;
        mutable std::optional<cxxopts::ParseResult> _parsed_cli_options;

        LM_NODISCARD const cxxopts::ParseResult &_parse_result() const noexcept;

        LM_NODISCARD const luminous_fs::path &_runtime_dir() noexcept;

        LM_NODISCARD const luminous_fs::path &_working_dir() noexcept;

        LM_NODISCARD const luminous_fs::path &_input_dir() noexcept;

        static bool _create_folder_if_necessary(const luminous_fs::path &path) noexcept;

    public:
        Context(int argc, char *argv[]);

        ~Context() noexcept;

        bool has_scene() {
            return _parse_result().count("scene") > 0;
        }

        LM_NODISCARD std::string load_cu_file(luminous_fs::path fn);

        LM_NODISCARD bool create_working_folder(const luminous_fs::path &name) noexcept;

        LM_NODISCARD bool create_cache_folder(const luminous_fs::path &name) noexcept;

        LM_NODISCARD luminous_fs::path include_path(const luminous_fs::path &name = {}) noexcept;

        LM_NODISCARD luminous_fs::path working_path(const luminous_fs::path &name = {}) noexcept;

        LM_NODISCARD luminous_fs::path runtime_path(const luminous_fs::path &name = {}) noexcept;

        LM_NODISCARD luminous_fs::path input_path(const luminous_fs::path &name = {}) noexcept;

        LM_NODISCARD luminous_fs::path cache_path(const luminous_fs::path &name = {}) noexcept;

        LM_NODISCARD luminous_fs::path scene_path() noexcept;

        LM_NODISCARD const luminous_fs::path &scene_file() noexcept;

        bool has_help_cmd() const noexcept {
            return _parse_result().count("help") > 0;
        }

        LM_NODISCARD bool use_gpu() noexcept {
            return device() == "cuda";
        }

        LM_NODISCARD bool show_window() noexcept;

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