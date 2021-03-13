//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "common.h"

#define HAVE__ALIGNED_MALLOC

namespace lstd {


    // memory_resource...
    namespace pmr {

        class memory_resource {
            static constexpr size_t max_align = alignof(std::max_align_t);

        public:
            virtual ~memory_resource();

            void *allocate(size_t bytes, size_t alignment = max_align) {
                if (bytes == 0)
                    return nullptr;
                return do_allocate(bytes, alignment);
            }

            void deallocate(void *p, size_t bytes, size_t alignment = max_align) {
                if (!p)
                    return;
                return do_deallocate(p, bytes, alignment);
            }

            bool is_equal(const memory_resource &other) const noexcept {
                return do_is_equal(other);
            }

        private:
            virtual void *do_allocate(size_t bytes, size_t alignment) = 0;

            virtual void do_deallocate(void *p, size_t bytes, size_t alignment) = 0;

            virtual bool do_is_equal(const memory_resource &other) const noexcept = 0;
        };

        inline bool operator==(const memory_resource &a, const memory_resource &b) noexcept {
            return a.is_equal(b);
        }

        inline bool operator!=(const memory_resource &a, const memory_resource &b) noexcept {
            return !(a == b);
        }

        memory_resource *new_delete_resource() noexcept;

        memory_resource *set_default_resource(memory_resource *r) noexcept;

        memory_resource *get_default_resource() noexcept;

        template<class Tp = std::byte>
        class polymorphic_allocator {
        public:
            using value_type = Tp;

            polymorphic_allocator() noexcept { _memory_resource = new_delete_resource(); }

            polymorphic_allocator(memory_resource *r) : _memory_resource(r) {}

            polymorphic_allocator(const polymorphic_allocator &other) = default;

            template<class U>
            polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept
                    : _memory_resource(other.resource()) {}

            polymorphic_allocator &operator=(const polymorphic_allocator &rhs) = delete;

            // member functions
            [[nodiscard]] Tp *allocate(size_t n) {
                return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
            }

            void deallocate(Tp *p, size_t n) { resource()->deallocate(p, n); }

            void *allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
                return resource()->allocate(nbytes, alignment);
            }

            void deallocate_bytes(void *p, size_t nbytes,
                                  size_t alignment = alignof(std::max_align_t)) {
                return resource()->deallocate(p, nbytes, alignment);
            }

            template<class T>
            T *allocate_object(size_t n = 1) {
                return static_cast<T *>(allocate_bytes(n * sizeof(T), alignof(T)));
            }

            template<class T>
            void deallocate_object(T *p, size_t n = 1) {
                deallocate_bytes(p, n * sizeof(T), alignof(T));
            }

            template<class T, class... Args>
            T *new_object(Args &&... args) {
                // NOTE: this doesn't handle constructors that throw exceptions...
                T *p = allocate_object<T>();
                construct(p, std::forward<Args>(args)...);
                return p;
            }

            template<class T>
            void delete_object(T *p) {
                destroy(p);
                deallocate_object(p);
            }

            template<class T, class... Args>
            void construct(T *p, Args &&... args) {
                ::new((void *) p) T(std::forward<Args>(args)...);
            }

            template<class T>
            void destroy(T *p) {
                p->~T();
            }

            memory_resource *resource() const { return _memory_resource; }

        private:
            memory_resource *_memory_resource;
        };

        template<class T1, class T2>
        bool operator==(const polymorphic_allocator<T1> &a,
                        const polymorphic_allocator<T2> &b) noexcept {
            return a.resource() == b.resource();
        }

        template<class T1, class T2>
        bool operator!=(const polymorphic_allocator<T1> &a,
                        const polymorphic_allocator<T2> &b) noexcept {
            return !(a == b);
        }

    }  // namespace pmr

    using Allocator = pmr::polymorphic_allocator<std::byte>;
}
