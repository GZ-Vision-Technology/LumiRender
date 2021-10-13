//
// Created by Zero on 2020/9/25.
//

#include "context.h"
//#include "rt_util/sampling.h"

namespace luminous {

    bool Context::_create_folder_if_necessary(const std::filesystem::path &path) noexcept {
        if (std::filesystem::exists(path)) {
            return true;
        }
        try {
            LUMINOUS_INFO("Creating folder: ", path);
            return std::filesystem::create_directories(path);
        } catch (const std::filesystem::filesystem_error &e) {
            LUMINOUS_WARNING("Failed to create folder ", path, ", reason: ", e.what());
        }
        return false;
    }

    bool Context::create_working_folder(const std::filesystem::path &name) noexcept {
        return _create_folder_if_necessary(working_path(name));
    }

    bool Context::create_cache_folder(const std::filesystem::path &name) noexcept {
        return _create_folder_if_necessary(cache_path(name));
    }

    std::filesystem::path Context::include_path(const std::filesystem::path &name) noexcept {
        return _runtime_dir() / "include" / name;
    }

    std::filesystem::path Context::working_path(const std::filesystem::path &name) noexcept {
        return _working_dir() / name;
    }

    std::filesystem::path Context::runtime_path(const std::filesystem::path &name) noexcept {
        return _runtime_dir() / name;
    }

    std::filesystem::path Context::cache_path(const std::filesystem::path &name) noexcept {
        return _working_dir() / "cache" / name;
    }

    std::filesystem::path Context::input_path(const std::filesystem::path &name) noexcept {
        return _input_dir() / name;
    }

    std::filesystem::path Context::scene_path() noexcept {
        return scene_file().parent_path();
    }


    Context::~Context() noexcept {
        for (auto &&module_item : _loaded_modules) {
            destroy_dynamic_module(module_item.second);
        }
    }

    Context::Context(int argc, char *argv[])
            : _argc{argc},
              _argv{const_cast<const char **>(argv)},
              _cli_options{std::filesystem::path{argv[0]}.filename().string()} {

        _cli_options.add_options()
                ("d, device", "Select compute device: cuda or cpu",
                        cxxopts::value<std::string>()->default_value("cuda"))
                ("r, runtime-dir", "Specify runtime directory",
                 cxxopts::value<std::filesystem::path>()->default_value(
                         std::filesystem::canonical(argv[0]).parent_path().parent_path().string()))
                ("w,working-dir", "Specify working directory",
                 cxxopts::value<std::filesystem::path>()->default_value(
                         std::filesystem::canonical(std::filesystem::current_path()).string()))
                ("c, clear-cache", "Clear cached", cxxopts::value<bool>())
                ("m, mode", "run mode: cli or gui",cxxopts::value<std::string>()->default_value("cli"))
                ("t, thread-num", "the num of threads to render", cxxopts::value<std::string>()->default_value("0"))
                ("s, scene", "The scene to render,file name end with json", cxxopts::value<std::string>())
                ("positional", "Specify input file", cxxopts::value<std::string>())
                ("h,help", "Print usage");

    }

    const cxxopts::ParseResult &Context::_parse_result() const noexcept {
        if (!_parsed_cli_options.has_value()) {
            _cli_options.parse_positional("positional");
            _parsed_cli_options.emplace(
                    _cli_options.parse(const_cast<int &>(_argc), const_cast<const char **&>(_argv)));
        }
        return *_parsed_cli_options;
    }

    const std::filesystem::path &Context::_runtime_dir() noexcept {
        if (_run_dir.empty()) {
            _run_dir = std::filesystem::canonical(_parse_result()["runtime-dir"].as<std::filesystem::path>());
            LUMINOUS_EXCEPTION_IF(!std::filesystem::exists(_run_dir) || !std::filesystem::is_directory(_run_dir),
                                  "Invalid runtime directory: ", _run_dir);
            LUMINOUS_INFO("Runtime directory: ", _run_dir);
        }
        return _run_dir;
    }

    const std::filesystem::path &Context::_working_dir() noexcept {
        if (_work_dir.empty()) {
            _work_dir = std::filesystem::canonical(_parse_result()["working-dir"].as<std::filesystem::path>());
            LUMINOUS_EXCEPTION_IF(!std::filesystem::exists(_work_dir) || !std::filesystem::is_directory(_work_dir),
                                  "Invalid working directory: ", _work_dir);
            std::filesystem::current_path(_work_dir);
            LUMINOUS_INFO("Working directory: ", _work_dir);
            auto cache_directory = _work_dir / "cache";
            if (_parse_result()["clear-cache"].as<bool>() && std::filesystem::exists(cache_directory)) {
                LUMINOUS_INFO("Removing cache directory: ", cache_directory);
                std::filesystem::remove_all(cache_directory);
            }
            LUMINOUS_EXCEPTION_IF(!_create_folder_if_necessary(cache_directory), "Failed to create cache directory: ",
                                  cache_directory);
        }
        return _work_dir;
    }

    const std::filesystem::path &Context::scene_file() noexcept {
        if (_scene_file.empty()) {
            _scene_file = std::filesystem::canonical(_parse_result()["scene"].as<std::string>());
        }
        return _scene_file;
    }

    std::string Context::device() noexcept {
        if (_device.empty()) {
            _device = _parse_result()["device"].as<std::string>();
        }
        return _device;
    }

    const std::filesystem::path &Context::_input_dir() noexcept {
        if (_in_dir.empty()) {
            if (_parse_result().count("positional") == 0u) {
                LUMINOUS_WARNING("No positional CLI argument given, setting input directory to working directory: ",
                                 _working_dir());
            } else {
                _in_dir = std::filesystem::canonical(cli_positional_option()).parent_path();
                LUMINOUS_EXCEPTION_IF(!std::filesystem::exists(_in_dir) || !std::filesystem::is_directory(_in_dir),
                                      "Invalid input directory: ", _in_dir);
                LUMINOUS_INFO("Input directory: ", _in_dir);
            }
        }
        return _in_dir;
    }

    bool Context::show_window() noexcept {
        return _parse_result()["mode"].as<std::string>() == "gui";
    }

}// namespace luminous