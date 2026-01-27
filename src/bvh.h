#pragma once

#include "sceneStructs.h"
#include <vector>
#include <glm/glm.hpp>
#include <iostream>

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
    int leafThreshold = 16;
};

struct BVHStats {
    int nodeCount = 0;
    int leafCount = 0;
    int internalCount = 0;
    int maxDepth = 0;
    int totalTriangles = 0;

    int minLeafTris = INT_MAX;
    int maxLeafTris = 0;
};

struct BVHTraverseStats {
    unsigned long long nodeVisits;
    unsigned long long triTests;
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


void printBVHStats(const BVH& bvh);

void collectBVHStats(
    const BVH& bvh,
    int nodeIdx,
    int depth,
    BVHStats& stats
);
