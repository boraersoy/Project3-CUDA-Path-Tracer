#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "bvh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
	std::vector<AABB> aabbs; //mesh aabbs maybe change the name later
	std::vector<AABB> triangleAABBs; //triangle aabbs for octree construction
	BVH cpubvh;
    RenderState state;
};
