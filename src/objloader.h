#pragma once

#include <string>
#include <vector>
#include "sceneStructs.h"

// Load an OBJ file into triangle list. Returns true on success.
bool loadOBJ(const std::string &path, std::vector<Triangle> &outTriangles);