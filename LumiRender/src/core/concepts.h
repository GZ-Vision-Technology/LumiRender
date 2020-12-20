//
// Created by Zero on 2020/9/1.
//
#pragma once

#include "header.h"

LUMINOUS_BEGIN

inline namespace utility {

    class IObject {

    };

    struct Noncopyable {
        Noncopyable() = default;

        Noncopyable(const Noncopyable &) = delete;

        Noncopyable &operator=(const Noncopyable &) = delete;
    };

    template<typename T>
    class ResourceGuard : public Noncopyable {

    public:
        using Delete = std::function<void(T &)>;

    private:
        T *_data;
        Delete _delete;

    public:
        ResourceGuard(T *data, Delete del) noexcept : _data{data}, _delete{std::move(del)} {}

        ResourceGuard(ResourceGuard &&other) noexcept : _data{other._data}, _delete{std::move(other._delete)} {
            other._data = nullptr;
        }

        ResourceGuard &operator=(ResourceGuard &other) noexcept {
            _data = other._data;
            other._data = nullptr;
            _delete = std::move(other._delete);
        }

        ~ResourceGuard() noexcept { if (_data != nullptr) { _delete(*_data); }}
    };

    template<typename T, typename Delete>
    [[nodiscard]] inline auto guard_resource(T *data, Delete &&d) noexcept {
        return ResourceGuard<T>{data, std::forward<Delete>(d)};
    }

};

LUMINOUS_END

