//
// Created by Zero on 2021/2/10.
//


#pragma once

#include "../geometry/common.h"

namespace lstd {
    // Array2D Definition
    template <typename T>
    class Array2D {
    public:
        // Array2D Type Definitions
        using value_type = T;
        using iterator = value_type *;
        using const_iterator = const value_type *;
        using allocator_type = lstd::pmr::polymorphic_allocator<std::byte>;

        // Array2D Public Methods
        Array2D(allocator_type allocator = {}) : Array2D({{0, 0}, {0, 0}}, allocator) {}

        Array2D(const Box2i &extent, Allocator allocator = {})
                : extent(extent), allocator(allocator) {
            int n = extent.Area();
            values = allocator.allocate_object<T>(n);
            for (int i = 0; i < n; ++i)
                allocator.construct(values + i);
        }

        Array2D(const Box2i &extent, T def, allocator_type allocator = {})
                : Array2D(extent, allocator) {
            std::fill(begin(), end(), def);
        }
        template <typename InputIt,
                typename = typename std::enable_if_t<
                        !std::is_integral<InputIt>::value &&
                        std::is_base_of<
                                std::input_iterator_tag,
                                typename std::iterator_traits<InputIt>::iterator_category>::value>>
        Array2D(InputIt first, InputIt last, int nx, int ny, allocator_type allocator = {})
                : Array2D({{0, 0}, {nx, ny}}, allocator) {
            std::copy(first, last, begin());
        }
        Array2D(int nx, int ny, allocator_type allocator = {})
                : Array2D({{0, 0}, {nx, ny}}, allocator) {}
        Array2D(int nx, int ny, T def, allocator_type allocator = {})
                : Array2D({{0, 0}, {nx, ny}}, def, allocator) {}
        Array2D(const Array2D &a, allocator_type allocator = {})
                : Array2D(a.begin(), a.end(), a.xSize(), a.ySize(), allocator) {}

        ~Array2D() {
            int n = extent.Area();
            for (int i = 0; i < n; ++i)
                allocator.destroy(values + i);
            allocator.deallocate_object(values, n);
        }

        Array2D(Array2D &&a, allocator_type allocator = {})
                : extent(a.extent), allocator(allocator) {
            if (allocator == a.allocator) {
                values = a.values;
                a.extent = Box2i({0, 0}, {0, 0});
                a.values = nullptr;
            } else {
                values = allocator.allocate_object<T>(extent.Area());
                std::copy(a.begin(), a.end(), begin());
            }
        }
        Array2D &operator=(const Array2D &a) = delete;

        Array2D &operator=(Array2D &&other) {
            if (allocator == other.allocator) {
                pstd::swap(extent, other.extent);
                pstd::swap(values, other.values);
            } else if (extent == other.extent) {
                int n = extent.Area();
                for (int i = 0; i < n; ++i) {
                    allocator.destroy(values + i);
                    allocator.construct(values + i, other.values[i]);
                }
                extent = other.extent;
            } else {
                int n = extent.Area();
                for (int i = 0; i < n; ++i)
                    allocator.destroy(values + i);
                allocator.deallocate_object(values, n);

                int no = other.extent.Area();
                values = allocator.allocate_object<T>(no);
                for (int i = 0; i < no; ++i)
                    allocator.construct(values + i, other.values[i]);
            }
            return *this;
        }

        XPU
                T &operator[](Point2i p) {
            DCHECK(InsideExclusive(p, extent));
            p.x -= extent.pMin.x;
            p.y -= extent.pMin.y;
            return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
        }
        XPU T &operator()(int x, int y) { return (*this)[{x, y}]; }

        XPU
        const T &operator()(int x, int y) const { return (*this)[{x, y}]; }
        XPU
        const T &operator[](Point2i p) const {
            DCHECK(InsideExclusive(p, extent));
            p.x -= extent.pMin.x;
            p.y -= extent.pMin.y;
            return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
        }

        XPU
        int size() const { return extent.Area(); }
        XPU
        int xSize() const { return extent.pMax.x - extent.pMin.x; }
        XPU
        int ySize() const { return extent.pMax.y - extent.pMin.y; }

        XPU
                iterator begin() { return values; }
        XPU
                iterator end() { return begin() + size(); }

        XPU
                const_iterator begin() const { return values; }
        XPU
                const_iterator end() const { return begin() + size(); }

        XPU
        operator pstd::span<T>() { return pstd::span<T>(values, size()); }
        XPU
        operator pstd::span<const T>() const { return pstd::span<const T>(values, size()); }

        std::string ToString() const {
            std::string s = StringPrintf("[ Array2D extent: %s values: [", extent);
            for (int y = extent.pMin.y; y < extent.pMax.y; ++y) {
                s += " [ ";
                for (int x = extent.pMin.x; x < extent.pMax.x; ++x) {
                    T value = (*this)(x, y);
                    s += StringPrintf("%s, ", value);
                }
                s += "], ";
            }
            s += " ] ]";
            return s;
        }

    private:
        // Array2D Private Members
        Box2i extent;
        Allocator allocator;
        T *values;
    };
}