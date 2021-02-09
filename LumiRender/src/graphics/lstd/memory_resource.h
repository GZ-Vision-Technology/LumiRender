//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "common.h"

namespace luminous {

    namespace lstd {

        class memory_resource {
            static constexpr size_t max_align = alignof(std::max_align_t);

        public:
            virtual ~memory_resource() {}

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
            virtual void *do_allocate(size_t bytes, size_t alignment) {
                return _aligned_malloc(bytes, alignment);
            }

            virtual void do_deallocate(void *ptr, size_t bytes, size_t alignment) {
                _aligned_free(ptr);
            }

            virtual bool do_is_equal(const memory_resource &other) const noexcept = 0;
        };

        inline bool operator==(const memory_resource &a, const memory_resource &b) noexcept {
            return a.is_equal(b);
        }

        inline bool operator!=(const memory_resource &a, const memory_resource &b) noexcept {
            return !(a == b);
        }

        struct pool_options {
            size_t max_blocks_per_chunk = 0;
            size_t largest_required_pool_block = 0;
        };

        class synchronized_pool_resource;

        class unsynchronized_pool_resource;

        // global memory resources
        memory_resource *new_delete_resource() noexcept;

        // TODO: memory_resource* null_memory_resource() noexcept;
        memory_resource *set_default_resource(memory_resource *r) noexcept;

        memory_resource *get_default_resource() noexcept;

        class monotonic_buffer_resource : public memory_resource {
        public:
            explicit monotonic_buffer_resource(memory_resource *upstream)
                    : _upstream_resource(upstream) {}

            monotonic_buffer_resource(size_t block_size, memory_resource *upstream)
                    : _block_size(block_size), _upstream_resource(upstream) {}

#if 0
            // TODO
            monotonic_buffer_resource(void *buffer, size_t buffer_size,
                                    memory_resource *upstream);
#endif

            monotonic_buffer_resource() : monotonic_buffer_resource(get_default_resource()) {}

            explicit monotonic_buffer_resource(size_t initial_size)
                    : monotonic_buffer_resource(initial_size, get_default_resource()) {}

#if 0
            // TODO
            monotonic_buffer_resource(void *buffer, size_t buffer_size)
                : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}
#endif

            monotonic_buffer_resource(const monotonic_buffer_resource &) = delete;

            ~monotonic_buffer_resource() { release(); }

            monotonic_buffer_resource operator=(const monotonic_buffer_resource &) = delete;

            void release() {
                for (const auto &block : _used_blocks)
                    _upstream_resource->deallocate(block.ptr, block.size);
                _used_blocks.clear();

                _upstream_resource->deallocate(_current_block.ptr, _current_block.size);
                _current_block = MemoryBlock();
            }

            memory_resource *upstream_resource() const { return _upstream_resource; }

        protected:
            void *do_allocate(size_t bytes, size_t align) override {
                if (bytes > _block_size) {
                    // We've got a big allocation; let the current block be so that
                    // smaller allocations have a chance at using up more of it.
                    _used_blocks.push_back(
                            MemoryBlock{_upstream_resource->allocate(bytes, align), bytes});
                    return _used_blocks.back().ptr;
                }

                if ((_current_block_pos % align) != 0)
                    _current_block_pos += align - (_current_block_pos % align);
                // DCHECK_EQ(0, _current_block_pos % align);

                if (_current_block_pos + bytes > _current_block.size) {
                    // Add current block to __used_blocks_ list
                    if (_current_block.size) {
                        _used_blocks.push_back(_current_block);
                        _current_block = {};
                    }

                    _current_block = {
                            _upstream_resource->allocate(_block_size, alignof(std::max_align_t)),
                            _block_size};
                    _current_block_pos = 0;
                }

                void *ptr = (char *) _current_block.ptr + _current_block_pos;
                _current_block_pos += bytes;
                return ptr;
            }

            void do_deallocate(void *p, size_t bytes, size_t alignment) override {
                // no-op
            }

            bool do_is_equal(const memory_resource &other) const noexcept override {
                return this == &other;
            }

        private:
            struct MemoryBlock {
                void *ptr = nullptr;
                size_t size = 0;
            };

            memory_resource *_upstream_resource;
            size_t _block_size = 256 * 1024;
            MemoryBlock _current_block;
            size_t _current_block_pos = 0;
            // TODO: should use the memory_resource for this list's allocations...
            std::list<MemoryBlock> _used_blocks;
        };

        template<class Tp = std::byte>
        class polymorphic_allocator {

        private:
            memory_resource *_memory_resource;

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

            // polymorphic_allocator select_on_container_copy_construction() const;

            memory_resource *resource() const { return _memory_resource; }

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


    } // luminous::lstd

    using Allocator = lstd::polymorphic_allocator<std::byte>;

} // luminous