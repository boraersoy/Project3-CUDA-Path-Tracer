#pragma once

#include "sceneStructs.h"

struct OctreeNode {
    AABB bounds;
    int children[8];   // -1 if leaf
    int triStart;
    int triCount;
};

struct Octree {
    std::vector<OctreeNode> nodes;
    std::vector<int> triIndices; // indices into triangle array
    int maxDepth;
    int leafThreshold;
};

void computeChildAABB(
    const AABB& parentAABB,
    int childIdx,
    AABB& outchildAABB
);

bool aabbOverlap(const AABB& a,
    const AABB& b
);

void buildNode(
    Octree& tree,
    int nodeIdx,
    int depth,
    const std::vector<AABB>& triBounds
);

void buildOctree(
    Octree& tree,
    std::vector<AABB>& triBounds,
    const AABB& sceneAABB
);

void printOctreeStats(const Octree& tree);

void validateOctree(const Octree& tree);