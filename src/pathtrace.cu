#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <chrono>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "objloader.h"
#include "bvh.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "denoise.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_BOUNCE 0
#define BBVH 0
#define AntiAliasing 0

__host__ __device__ float rand01(int x, int y, int iter) {
	unsigned int seed =
		x * 1973 +
		y * 9277 +
		iter * 26699 +
		89173;

	seed = (seed << 13) ^ seed;
	return 1.0f - ((seed * (seed * seed * 15731u + 789221u) + 1376312589u)
		& 0x7fffffff) / 1073741824.0f;
}




void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}
enum MaterialType {
	MAT_DIFFUSE = 0,
	MAT_SPECULAR = 1,
	MAT_EMISSIVE = 2
};
struct MaterialKeyFunctor {
	Material* materials;

	__host__ __device__
		int operator()(const ShadeableIntersection& si) const {
		if (si.t <= 0.0f) return 3;

		const Material& m = materials[si.materialId];
		if (m.emittance > 0.0f) return 2;
		if (m.specular.exponent > 0.0f) return 1;
		return 0;
	}
};

struct PathSegmentActive {
	__host__ __device__
		bool operator()(const PathSegment& p) const {
		return p.remainingBounces > 0;
	}
};

struct PathSegmentTerminated {
	__host__ __device__
		bool operator()(const PathSegment& p) const {
		return p.remainingBounces <= 0;
	}
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialKeys = nullptr;
static Triangle* dev_triangle = nullptr;
int num_triangles = 1;
static BVHNode* dev_bvhNodes = nullptr;
int* dev_triIndices = nullptr;
BVHTraverseStats* dev_bvhStats = nullptr;


//denoising
static GBufferPixel* dev_gBuffer = NULL;
static glm::ivec2* dev_offset = NULL;
static float* dev_kernel = NULL;
static glm::vec3* dev_denoiseBufferIn = NULL;
static glm::vec3* dev_denoiseBufferOut = NULL;

glm::ivec2 offset[25];
const float kernel[25] = {
	1.f / 256, 1.f / 64, 3.f / 128, 1.f / 64, 1.f / 256,
	1.f / 64, 1.f / 16, 3.f / 32, 1.f / 16, 1.f / 64,
	3.f / 128, 3.f / 32, 9.f / 64, 3.f / 32, 3.f / 128,
	1.f / 64, 1.f / 16, 3.f / 32, 1.f / 16, 1.f / 64,
	1.f / 256, 1.f / 64, 3.f / 128, 1.f / 64, 1.f / 256
};




#if CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_firstBounceIntersections = nullptr;
bool first_bounce_cached = false;
#endif

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
#if SORT_BY_MATERIAL

	cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));
#endif
#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_firstBounceIntersections, pixelcount * sizeof(ShadeableIntersection));
	first_bounce_cached = false;
#endif
	//create a single hardcoded triangle

	num_triangles = (int)hst_scene->triangles.size();


	cudaMalloc(&dev_triangle, num_triangles * sizeof(Triangle));
	cudaMemcpy(dev_triangle, hst_scene->triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);

	//allocate device memory for single mesh AABB
#if BBVH
	BVH& bvh = hst_scene->cpubvh;

	std::vector<int>& cpuTriIndices = bvh.triangleIndices;
	cudaMalloc(
		&dev_bvhNodes,
		bvh.nodes.size() * sizeof(BVHNode)
	);
	cudaMemcpy(
		dev_bvhNodes,
		bvh.nodes.data(),
		bvh.nodes.size() * sizeof(BVHNode),
		cudaMemcpyHostToDevice
	);

	cudaMalloc(
		&dev_triIndices,
		cpuTriIndices.size() * sizeof(int)
	);

	cudaMemcpy(
		dev_triIndices,
		cpuTriIndices.data(),
		cpuTriIndices.size() * sizeof(int),
		cudaMemcpyHostToDevice
	);
#endif BBVH
	cudaMalloc(&dev_bvhStats, sizeof(BVHTraverseStats));

	//denoising
	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));



	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_denoiseBufferIn, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_denoiseBufferOut, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_offset, 25 * sizeof(glm::ivec2));
	for (int i = 0, int y = 0; y < 5; ++y) { // read array from left to right, top to bottom
		for (int x = 0; x < 5; ++x) {
			offset[i++] = glm::ivec2(x - 2, y - 2);
		}
	}
	cudaMemcpy(dev_offset, offset, 25 * sizeof(glm::ivec2), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_kernel, 25 * sizeof(float));
	cudaMemcpy(dev_kernel, kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
#if SORT_BY_MATERIAL
	cudaFree(dev_materialKeys);
#endif
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_firstBounceIntersections);
	first_bounce_cached = false;
#endif
	// TODO: clean up any extra device memory you created
	cudaFree(dev_triangle);
#if BBVH
	cudaFree(dev_bvhNodes);
	cudaFree(dev_triIndices);
#endif

	//denoising
	cudaFree(dev_gBuffer);
	cudaFree(dev_denoiseBufferIn);
	cudaFree(dev_denoiseBufferOut);
	cudaFree(dev_offset);
	cudaFree(dev_kernel);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing
- add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if AntiAliasing
		float jx = rand01(x, y, iter);
		float jy = rand01(x + 17, y + 13, iter);


		float px = (float)x + jx;
		float py = (float)y + jy;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)px - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)py - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, Triangle* triangles
	, int geoms_size
	, ShadeableIntersection* intersections
	, BVHNode* dev_bvhNodes
	, int* dev_triIndices
	, BVHTraverseStats* dev_bvhStats
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t; // world space intersection distance along ray
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1; // what object this intersection hit. Index should be index in dev_geoms
		bool outside = true; // if it hit outer surface of object or not. Not sure what to do if false

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;


		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH) { // TODO: add more intersection tests here... triangle? metaball? CSG?

#if BBVH
				t = traverseBVH(
					dev_bvhNodes,
					dev_triIndices,
					triangles,
					geom,
					pathSegment.ray,
					tmp_intersect,
					tmp_normal,
					outside,
					dev_bvhStats
				);
#else 
				t = triangleMeshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside, dev_bvhStats);
#endif
			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;

			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f; // GUESSING: -1 means did not hit (raymarch didn't go anywhere?)
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}






__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			thrust::default_random_engine rng =
				makeSeededRandomEngine(iter, idx, depth);

			Material material = materials[intersection.materialId];

			scatterRay(
				pathSegments[idx],
				intersection,
				material,
				rng
			);
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}




// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}


__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		auto& intersect = shadeableIntersections[idx];
		gBuffer[idx].t = intersect.t;
		gBuffer[idx].normal = intersect.surfaceNormal;
		gBuffer[idx].position = intersect.t *
			pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
	}
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer[index].t * 256.0;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;
	}
}

__global__ void nbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 normal = gBuffer[index].normal;
		glm::vec3 normalized = glm::clamp(glm::abs(256.f * normal), 0.f, 255.f);
		uchar4 color = make_uchar4(normalized.x, normalized.y, normalized.z, 0);
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void pbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 position = gBuffer[index].position;
		glm::vec3 normalized = glm::clamp(glm::abs(256.f * position * 0.1f), 0.f, 255.f);
		uchar4 color = make_uchar4(normalized.x, normalized.y, normalized.z, 0);
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showNBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	nbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showPBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	pbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

__global__ void kernInitDenoiseBuffer(glm::vec3* image, glm::ivec2 resolution, float iter, glm::vec3* inputImage) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((x < resolution.x && y < resolution.y)) {
		int index = x + (y * resolution.x);
		inputImage[index] = image[index] / iter;
	}
}
__device__ __forceinline__ float getWeight(glm::vec3 v1, glm::vec3 v2, float sigma) {
	glm::vec3 t = v1 - v2;
	float dist_squared = glm::dot(t, t);
	return glm::min(exp(-dist_squared / (sigma * sigma)), 1.0f);
}
__global__ void atrousFilter(
	glm::vec3* inputImage,
	GBufferPixel* gBuffer,
	glm::ivec2 resolution,
	int stepWidth,
	float c_phi,
	float n_phi,
	float p_phi,
	const float* kernel,
	const glm::ivec2* offset,
	glm::vec3* outputImage

) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		auto& color = inputImage[index];
		auto& position = gBuffer[index].position;
		auto& normal = gBuffer[index].normal;

		glm::vec3 sum = glm::vec3(0.0f);
		float cum_w = 0.0f;

		for (int i = 0; i < 25; i++) {
			glm::ivec2 uv = glm::ivec2(x, y) + offset[i] * stepWidth;

			// Clamp to image boundaries
			uv.x = glm::clamp(uv.x, 0, resolution.x - 1);
			uv.y = glm::clamp(uv.y, 0, resolution.y - 1);

			int uv1d = uv.x + uv.y * resolution.x;

			auto& neighbor_color = inputImage[uv1d];
			auto& neighbor_position = gBuffer[uv1d].position;
			auto& neighbor_normal = gBuffer[uv1d].normal;

			// Calculate edge-aware weights (or skip if sigma is 0)
			float c_w = (c_phi > 0.0f) ? getWeight(color, neighbor_color, c_phi) : 1.0f;
			float n_w = (n_phi > 0.0f) ? getWeight(normal, neighbor_normal, n_phi) : 1.0f;
			float p_w = (p_phi > 0.0f) ? getWeight(position, neighbor_position, p_phi) : 1.0f;

			float weight = c_w * n_w * p_w * kernel[i];
			sum += weight * neighbor_color;
			cum_w += weight;
		}
		outputImage[index] = sum / cum_w;

	}
}

void denoise(uchar4* image, int filterSize, int iter, float c_phi,
	float n_phi, float p_phi) {
	const Camera& cam = hst_scene->state.camera;
	glm::vec2 resolution = cam.resolution;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	const int pixelcount = resolution.x * resolution.y;

	// Copy input image to buffer A (normalized by iteration count)
	kernInitDenoiseBuffer << <blocksPerGrid2d, blockSize2d >> > (dev_image, resolution, iter, dev_denoiseBufferIn);


	//number of iterations
	int numIterations = glm::clamp(filterSize / 20, 1, 5);
	// Apply A-trous filter multiple times with increasing step size
	for (int i = 0; i < numIterations; i++) {
		int stepWidth = 1 << i;  // 2^i: 1, 2, 4, 8, 16...

		atrousFilter << <blocksPerGrid2d, blockSize2d >> > (dev_denoiseBufferIn, dev_gBuffer, resolution,
			stepWidth, c_phi, n_phi, p_phi, dev_kernel, dev_offset, dev_denoiseBufferOut);
		cudaDeviceSynchronize();
		checkCUDAError("atrous filter iteration");

		// Swap buffers for next iteration
		std::swap(dev_denoiseBufferIn, dev_denoiseBufferOut);
	}

	// Copy result back to dev_image (with iteration scaling for display)
	//cudaMemcpy(dev_image, dev_denoiseBufferIn, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (image, resolution, 1, dev_denoiseBufferIn);
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

}






/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	//const int traceDepth = 2; hardcoded for easier testing
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	
	// Start timing the entire iteration
	auto iterationStartTime = std::chrono::high_resolution_clock::now();

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		
		// Start timing this depth level
		auto depthStartTime = std::chrono::high_resolution_clock::now();

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
		if (depth == 0) {
			if (!first_bounce_cached) {
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth,
					num_paths,
					dev_paths,
					dev_geoms,
					dev_triangle,
					hst_scene->geoms.size(),
					dev_intersections,
					dev_bvhNodes,
					dev_triIndices,
					dev_bvhStats
					);

				// cache first bounce
				cudaMemcpy(
					dev_firstBounceIntersections,
					dev_intersections,
					pixelcount * sizeof(ShadeableIntersection),
					cudaMemcpyDeviceToDevice
				);

				first_bounce_cached = true;
			}
			else {
				// reuse cached first bounce
				cudaMemcpy(
					dev_intersections,
					dev_firstBounceIntersections,
					pixelcount * sizeof(ShadeableIntersection),
					cudaMemcpyDeviceToDevice
				);
			}
		}
		else {
			cudaMemset(dev_bvhStats, 0, sizeof(BVHTraverseStats));
			
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				dev_triangle,
				hst_scene->geoms.size(),
				dev_intersections,
				dev_bvhNodes,
				dev_triIndices,
				dev_bvhStats
				);
		}
#else
		cudaMemset(dev_bvhStats, 0, sizeof(BVHTraverseStats));

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			dev_triangle,
			hst_scene->geoms.size(),
			dev_intersections,
			dev_bvhNodes,
			dev_triIndices,
			dev_bvhStats
			);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}
		depth++;
		//print the bvh traversal stats
		BVHTraverseStats hst_bvhStats;
		//cudaMemcpy(&hst_bvhStats, dev_bvhStats, sizeof(BVHTraverseStats), cudaMemcpyDeviceToHost);
		//printf("Depth %d: BVH Node Visits: %llu, Triangle Tests: %llu, Triangles Per Ray: %llu\n",
		//	depth,
		//	hst_bvhStats.nodeVisits,
		//	hst_bvhStats.triTests,
		//	hst_bvhStats.triTests / num_paths
		//);




		//cache the first bounce (intersection) 


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
		MaterialKeyFunctor keyFn;
		keyFn.materials = dev_materials;

		thrust::transform(
			thrust::device,
			dev_intersections,
			dev_intersections + num_paths,
			dev_materialKeys,
			keyFn
		);

		// 2) zip paths + intersections
		auto zipped_begin = thrust::make_zip_iterator(
			thrust::make_tuple(dev_paths, dev_intersections)
		);

		// 3) stable sort by material
		thrust::stable_sort_by_key(
			thrust::device,
			dev_materialKeys,
			dev_materialKeys + num_paths,
			zipped_begin
		);
#endif
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth
			);

		checkCUDAError("shade material");


		//stream compaction using stable_partition
		thrust::device_ptr<PathSegment> path_ptr =
			thrust::device_pointer_cast(dev_paths);

		auto new_end = thrust::stable_partition(
			thrust::device,
			path_ptr,
			path_ptr + num_paths,
			PathSegmentActive()
		);

		num_paths = new_end - path_ptr;
		
		// End timing for this depth level
		cudaDeviceSynchronize();  // Ensure all GPU work is complete before measuring time
		auto depthEndTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> depthDuration = depthEndTime - depthStartTime;
		
		// Store timing for this depth
		if (guiData != NULL && depth - 1 < GuiDataContainer::MAX_DEPTH) {
			guiData->DepthTimings[depth - 1] = depthDuration.count();
		}
		
		// Print depth timing to console
	//	printf("Depth %d: Time = %.3f ms, Active paths = %d\n", depth - 1, depthDuration.count(), num_paths);

		if (num_paths == 0 || depth >= traceDepth) {
			iterationComplete = true;
		}


		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();  // Ensure all work is complete
	
	// Calculate total iteration time
	auto iterationEndTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> iterationDuration = iterationEndTime - iterationStartTime;
	
	if (guiData != NULL) {
		guiData->TotalIterationTime = iterationDuration.count();
	}
	
	//// Print total iteration timing summary to console
	//printf("----------------------------------------\n");
	//printf("Iteration %d complete: Total time = %.3f ms\n", iter, iterationDuration.count());
	//printf("----------------------------------------\n\n");

	checkCUDAError("pathtrace");

}

