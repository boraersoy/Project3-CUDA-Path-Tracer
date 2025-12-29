#include "bvh.h"
#include <algorithm>
#include <cfloat>

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------
static inline glm::vec3 aabbCenter(const AABB& box) {
    return 0.5f * (box.min + box.max);
}

static inline AABB unionAABB(const AABB& a, const AABB& b) {
    AABB out;
    out.min = glm::min(a.min, b.min);
    out.max = glm::max(a.max, b.max);
    return out;
}

// ------------------------------------------------------------
// Entry point
// ------------------------------------------------------------
void buildBVH(
    BVH& bvh,
    const std::vector<AABB>& triangleAABBs,
    const AABB& rootBounds
) {
    const int triCount = static_cast<int>(triangleAABBs.size());

    bvh.nodes.clear();
    bvh.triangleIndices.clear();

    // Initialize triangle index list
    bvh.triangleIndices.resize(triCount);
    for (int i = 0; i < triCount; i++) {
        bvh.triangleIndices[i] = i;
    }

    // Create root node
    BVHNode root;
    root.bounds = rootBounds;
    root.left = -1;
    root.right = -1;
    root.triStart = 0;
    root.triCount = triCount;

    bvh.nodes.push_back(root);

    // Build recursively
    buildBVHNode(bvh, 0, 0, triangleAABBs);
}

// ------------------------------------------------------------
// Recursive BVH builder
// ------------------------------------------------------------
void buildBVHNode(
    BVH& bvh,
    int nodeIdx,
    int depth,
    const std::vector<AABB>& triangleAABBs
) {
    // Copy values we need (NOT reference!)
    int triStart = bvh.nodes[nodeIdx].triStart;
    int triCount = bvh.nodes[nodeIdx].triCount;
    AABB bounds = bvh.nodes[nodeIdx].bounds;

    if (triCount <= bvh.leafThreshold || depth >= bvh.maxDepth)
        return;

    int axis = longestAxis(bounds);

    int start = triStart;
    int end = start + triCount;
    int mid = start + triCount / 2;

    // Correct comparator
    auto centroidCmp = [&](int ia, int ib) {
        int a = bvh.triangleIndices[ia];
        int b = bvh.triangleIndices[ib];
        return aabbCenter(triangleAABBs[a])[axis] <
            aabbCenter(triangleAABBs[b])[axis];
        };

    std::nth_element(
        bvh.triangleIndices.begin() + start,
        bvh.triangleIndices.begin() + mid,
        bvh.triangleIndices.begin() + end,
        centroidCmp
    );

    if (mid <= start || mid >= end)
        return;

    // Create left child
    BVHNode left;
    left.left = -1;
    left.right = -1;
    left.triStart = start;
    left.triCount = mid - start;
    left.bounds = computeNodeBounds(
        bvh, left.triStart, left.triCount, triangleAABBs
    );

    // Create right child
    BVHNode right;
    right.left = -1;
    right.right = -1;
    right.triStart = mid;
    right.triCount = end - mid;
    right.bounds = computeNodeBounds(
        bvh, right.triStart, right.triCount, triangleAABBs
    );

    int leftIdx = (int)bvh.nodes.size();
    bvh.nodes.push_back(left);

    int rightIdx = (int)bvh.nodes.size();
    bvh.nodes.push_back(right);

    // SAFE write-back after push_back
    bvh.nodes[nodeIdx].left = leftIdx;
    bvh.nodes[nodeIdx].right = rightIdx;

    buildBVHNode(bvh, leftIdx, depth + 1, triangleAABBs);
    buildBVHNode(bvh, rightIdx, depth + 1, triangleAABBs);
}


// ------------------------------------------------------------
// Compute bounds for a node
// ------------------------------------------------------------
AABB computeNodeBounds(
    const BVH& bvh,
    int triStart,
    int triCount,
    const std::vector<AABB>& triangleAABBs
) {
    AABB bounds;
    bounds.min = glm::vec3(FLT_MAX);
    bounds.max = glm::vec3(-FLT_MAX);


    for (int i = 0; i < triCount; i++) {
        int triIdx = bvh.triangleIndices[triStart + i];
        bounds = unionAABB(bounds, triangleAABBs[triIdx]);
    }

    return bounds;
}

// ------------------------------------------------------------
// Longest axis selection
// ------------------------------------------------------------
int longestAxis(const AABB& box) {
    glm::vec3 extent = box.max - box.min;

    if (extent.x > extent.y && extent.x > extent.z) return 0;
    if (extent.y > extent.z)                       return 1;
    return 2;
}

