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

// TODO
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
                    : upstreamResource(upstream) {}

            monotonic_buffer_resource(size_t blockSize, memory_resource *upstream)
                    : blockSize(blockSize), upstreamResource(upstream) {}

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
                for (const auto &block : usedBlocks)
                    upstreamResource->deallocate(block.ptr, block.size);
                usedBlocks.clear();

                upstreamResource->deallocate(currentBlock.ptr, currentBlock.size);
                currentBlock = MemoryBlock();
            }

            memory_resource *upstream_resource() const { return upstreamResource; }

        protected:
            void *do_allocate(size_t bytes, size_t align) override {
                if (bytes > blockSize) {
                    // We've got a big allocation; let the current block be so that
                    // smaller allocations have a chance at using up more of it.
                    usedBlocks.push_back(
                            MemoryBlock{upstreamResource->allocate(bytes, align), bytes});
                    return usedBlocks.back().ptr;
                }

                if ((currentBlockPos % align) != 0)
                    currentBlockPos += align - (currentBlockPos % align);
                DCHECK_EQ(0, currentBlockPos % align);

                if (currentBlockPos + bytes > currentBlock.size) {
                    // Add current block to _usedBlocks_ list
                    if (currentBlock.size) {
                        usedBlocks.push_back(currentBlock);
                        currentBlock = {};
                    }

                    currentBlock = {
                            upstreamResource->allocate(blockSize, alignof(std::max_align_t)),
                            blockSize};
                    currentBlockPos = 0;
                }

                void *ptr = (char *) currentBlock.ptr + currentBlockPos;
                currentBlockPos += bytes;
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

            memory_resource *upstreamResource;
            size_t blockSize = 256 * 1024;
            MemoryBlock currentBlock;
            size_t currentBlockPos = 0;
            // TODO: should use the memory_resource for this list's allocations...
            std::list<MemoryBlock> usedBlocks;
        };

        template<class Tp = std::byte>
        class polymorphic_allocator {
        public:
            using value_type = Tp;

            polymorphic_allocator() noexcept { memoryResource = new_delete_resource(); }

            polymorphic_allocator(memory_resource *r) : memoryResource(r) {}

            polymorphic_allocator(const polymorphic_allocator &other) = default;

            template<class U>
            polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept
                    : memoryResource(other.resource()) {}

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

            memory_resource *resource() const { return memoryResource; }

        private:
            memory_resource *memoryResource;
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

namespace lstd {
    namespace pmr {

        memory_resource::~memory_resource() {}

        class NewDeleteResource : public memory_resource {
            void *do_allocate(size_t size, size_t alignment) {
#if defined(HAVE__ALIGNED_MALLOC)
                return _aligned_malloc(size, alignment);
#elif defined(HAVE_POSIX_MEMALIGN)
                void *ptr;
        if (alignment < sizeof(void *))
            return malloc(size);
        if (posix_memalign(&ptr, alignment, size) != 0)
            ptr = nullptr;
        return ptr;
#else
                return memalign(alignment, size);
#endif
            }

            void do_deallocate(void *ptr, size_t bytes, size_t alignment) {
                if (ptr == nullptr)
                    return;
#if defined(HAVE__ALIGNED_MALLOC)
                _aligned_free(ptr);
#else
                free(ptr);
#endif
            }

            bool do_is_equal(const memory_resource &other) const noexcept {
                return this == &other;
            }
        };

        static NewDeleteResource ndr;

        memory_resource *new_delete_resource() noexcept {
            return &ndr;
        }

        static memory_resource *defaultMemoryResource = new_delete_resource();

        memory_resource *set_default_resource(memory_resource *r) noexcept {
            memory_resource *orig = defaultMemoryResource;
            defaultMemoryResource = r;
            return orig;
        }

        memory_resource *get_default_resource() noexcept {
            return defaultMemoryResource;
        }

    }  // namespace pmr

} // lstd