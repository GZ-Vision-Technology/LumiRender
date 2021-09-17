//
// Created by Zero on 17/09/2021.
//


#pragma once

#include "base_libs/math/interval.h"
#include <vector>
#include <queue>

namespace luminous {
    inline namespace refl {

        class Object;

        struct MemorySegment {
            PtrInterval host_interval{};
            uint64_t device_ptr{0};

            uint64_t length() {
                host_interval.span();
            }

        };

        struct Node {
            int memory_segment_index{-1};
            uint64_t start{0}, end{0};
            std::vector<Node> children;

            void add_children(const Node &node) {
                children.push_back(node);
            }
        };

        struct ReferenceTree {
        private:
            Node _root;

            template<typename F>
            void depth_first_traverse(Node *node, const F &f) {
                f(node);
                for (Node &child : node->children) {
                    depth_first_traverse(&child, f);
                }
            }

        public:
            template<typename F>
            void depth_first_traverse(const F &f) {
                depth_first_traverse(&_root, f);
            }

            template<typename F>
            void breadth_first_traverse(const F &f) {
                std::queue<Node> node_queue;
                node_queue.push(_root);

                do {
                    Node &node = node_queue.back();
                    node_queue.pop();
                    for (Node &child : node.children) {
                        node_queue.push(child);
                    }
                    f(&node);
                } while (!node_queue.empty());
            }
        };

        class PtrMapper {
        private:
            std::vector<MemorySegment> _memory_segments;
            std::vector<ReferenceTree> _trees;
        public:
            PtrMapper() = default;

            void add_tree(Object *object);
        };
    }
}