#include "octree.h"
#include <iostream>

void computeChildAABB(
	const AABB& parentAABB,
	int childIdx,
	AABB& outchildAABB
) {
	glm::vec3 center = (parentAABB.min + parentAABB.max) * 0.5f;
	outchildAABB.min = parentAABB.min;
	outchildAABB.max = parentAABB.max;

	if (childIdx & 1) outchildAABB.min.x = center.x;
	else outchildAABB.max.x = center.x;

	if (childIdx & 2) outchildAABB.min.y = center.y;
	else outchildAABB.max.y = center.y;

	if (childIdx & 4) outchildAABB.min.z = center.z;
	else outchildAABB.max.z = center.z;
}

bool aabbOverlap(const AABB& a,
	const AABB& b
) {
	return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
		(a.min.y <= b.max.y && a.max.y >= b.min.y) &&
		(a.min.z <= b.max.z && a.max.z >= b.min.z);
}

void buildNode(
    Octree& tree,
    int nodeIdx,
    int depth,
    const std::vector<AABB>& triBounds
) {
    // Bounds check on node index (paranoia but safe)
    if (nodeIdx < 0 || nodeIdx >= (int)tree.nodes.size())
        return;

    OctreeNode& node = tree.nodes[nodeIdx];

    // Termination condition
    if (depth >= tree.maxDepth || node.triCount <= tree.leafThreshold)
        return;

    // Initialize children
    for (int i = 0; i < 8; i++)
        node.children[i] = -1;

    // Cache parent triangle range (CRITICAL)
    const int parentTriStart = node.triStart;
    const int parentTriCount = node.triCount;

    // Sanity check
    if (parentTriStart < 0 ||
        parentTriStart + parentTriCount >(int)tree.triIndices.size())
        return;

    // Subdivide
    for (int c = 0; c < 8; c++) {
        OctreeNode child;

        // Compute child bounds
        computeChildAABB(node.bounds, c, child.bounds);

        for (int i = 0; i < 8; i++)
            child.children[i] = -1;

        child.triStart = (int)tree.triIndices.size();
        child.triCount = 0;

        // Assign triangles using cached parent range
        for (int i = 0; i < parentTriCount; i++) {
            int triIdx = tree.triIndices[parentTriStart + i];

            // Triangle index sanity
            if (triIdx < 0 || triIdx >= (int)triBounds.size())
                continue;

            if (aabbOverlap(child.bounds, triBounds[triIdx])) {
                tree.triIndices.push_back(triIdx);
                child.triCount++;
            }
        }

        // Only add non-empty children
        if (child.triCount > 0) {
            node.children[c] = (int)tree.nodes.size();
            tree.nodes.push_back(child);
        }
    }

    // Clear parent triangle list (IMPORTANT)
    node.triStart = 0;
    node.triCount = 0;

    // Recurse
    for (int c = 0; c < 8; c++) {
        if (node.children[c] != -1) {
            buildNode(tree, node.children[c], depth + 1, triBounds);
        }
    }
}


void buildOctree(
    Octree& tree,
    std::vector<AABB>& triBounds,
    const AABB& sceneAABB
) {
    tree.nodes.clear();
    tree.triIndices.clear();
	tree.maxDepth = 8;
	tree.leafThreshold = 10;

    OctreeNode root;
	root.bounds = sceneAABB;
    printf("Root AABB min: %f %f %f\n", sceneAABB.min.x, sceneAABB.min.y, sceneAABB.min.z);
    printf("Root AABB max: %f %f %f\n", sceneAABB.max.x, sceneAABB.max.y, sceneAABB.max.z);
    root.triStart = 0;
	root.triCount = static_cast<int>(triBounds.size());
	// Initialize root triangle indices
    for (int i = 0; i < 8; i++) root.children[i] = -1;

	tree.nodes.push_back(root);

    for (int i = 0; i < root.triCount; i++) {
        tree.triIndices.push_back(i);
	}

	buildNode(tree, 0, 0, triBounds);
}

void validateOctree(const Octree& tree) {
    for (size_t i = 0; i < tree.nodes.size(); i++) {
        const OctreeNode& n = tree.nodes[i];

        if (n.triCount == 0) {
            bool hasChild = false;
            for (int c = 0; c < 8; c++) {
                if (n.children[c] != -1)
                    hasChild = true;
            }
     

            if (!hasChild) {
                std::cerr << "Empty leaf at node " << i << std::endl;
            }
        }
    }
}

void printOctreeStats(const Octree& tree) {
    int maxDepth = 0;
    int leafCount = 0;
    int totalLeafTris = 0;

    for (const auto& n : tree.nodes) {
        bool isLeaf = true;
        for (int c = 0; c < 8; c++)
            if (n.children[c] != -1)
                isLeaf = false;

        if (isLeaf) {
            leafCount++;
            totalLeafTris += n.triCount;
        }
    }

    std::cout << "Nodes: " << tree.nodes.size() << "\n";
    std::cout << "Leaves: " << leafCount << "\n";
    std::cout << "Avg tris/leaf: "
        << (leafCount ? float(totalLeafTris) / leafCount : 0)
        << "\n";
}
