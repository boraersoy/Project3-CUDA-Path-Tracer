#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "objloader.h"
#include <iostream>

bool loadOBJ(
    const std::string& path,
    std::vector<Triangle>& outTriangles,
	AABB& outAABB
) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    tinyobj::LoadObj(
        &attrib,
        &shapes,
        &materials,
        &warn,
        &err,
        path.c_str(),
        nullptr
        
    );

    if (!warn.empty()) std::cout << warn << std::endl;
    if (!err.empty())  std::cerr << err << std::endl;

    outAABB.min = glm::vec3(FLT_MAX);
    outAABB.max = glm::vec3(-FLT_MAX);

    for (const auto& shape : shapes) {
        size_t index_offset = 0;

        glm::vec3 minBounds(FLT_MAX);
        glm::vec3 maxBounds(-FLT_MAX);

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];

            // We only support triangles
            if (fv != 3) {
                index_offset += fv;
                continue;
            }


            Triangle tri;

            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                glm::vec3 pos(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                if (v == 0) tri.v0 = pos;
                if (v == 1) tri.v1 = pos;
                if (v == 2) tri.v2 = pos;

				minBounds = glm::min(minBounds, pos);
				maxBounds = glm::max(maxBounds, pos);
            }

            tri.normal = glm::normalize(
                glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0)
            );

            outTriangles.push_back(tri);

			

            index_offset += fv;
        }
        outAABB.min = minBounds;
        outAABB.max = maxBounds;
    }

    return true;
}
