#pragma once

#include "sceneStructs.h"
#include <vector>
#include <glm/glm.hpp>

struct BVHNode {
    AABB bounds;

    int left;      // index of left child (-1 if leaf)
    int right;     // index of right child (-1 if leaf)

    int triStart;  // start index in triangleIndices
    int triCount;  // number of triangles in this node
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int> triangleIndices;

    int maxDepth = 32;
    int leafThreshold = 4;
};

// Entry point
void buildBVH(
    BVH& bvh,
    const std::vector<AABB>& triangleAABBs,
    const AABB& rootBounds
);

// Recursive builder
void buildBVHNode(
    BVH& bvh,
    int nodeIdx,
    int depth,
    const std::vector<AABB>& triangleAABBs
);

AABB computeNodeBounds(
    const BVH& bvh,
    int triStart,
    int triCount,
    const std::vector<AABB>& triangleAABBs
);

int longestAxis(const AABB& box);

__host__ __device__
bool intersectAABB(
    const AABB& box,
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir
);