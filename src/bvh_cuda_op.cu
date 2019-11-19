/*
 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
 holder of all proprietary rights on this computer program.
 You can only use this computer program if you have closed
 a license agreement with MPG or you get the right to use the computer
 program from someone who is authorized to grant you that right.
 Any use of the computer program without a valid license is prohibited and
 liable to prosecution.

 Copyright©2019 Max-Planck-Gesellschaft zur Förderung
 der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
 for Intelligent Systems and the Max Planck Institute for Biological
 Cybernetics. All rights reserved.

 Contact: ps-license@tuebingen.mpg.de
*/
#include <torch/extension.h>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "aabb.hpp"
#include "defs.hpp"
#include "double_vec_ops.h"
#include "helper_math.h"
#include "math_utils.hpp"
#include "priority_queue.hpp"

#ifndef EPSILON
#define EPSILON 1e-16
#endif /* ifndef EPSILON */

// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

typedef unsigned int MortonCode;

template <typename T>
std::ostream &operator<<(std::ostream &os, const vec3<T> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<float> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<double> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

template <typename T> std::ostream &operator<<(std::ostream &os, vec3<T> x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

__host__ __device__ inline double3 fmin(const double3 &a, const double3 &b) {
  return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3 &a, const double3 &b) {
  return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
  __host__ __device__ int operator()(long2 vec) const {
    return vec.x >= 0 && vec.y >= 0;
  }
};

template <typename T> struct Triangle {
public:
  vec3<T> v0;
  vec3<T> v1;
  vec3<T> v2;

  __host__ __device__ Triangle(const vec3<T> &vertex0, const vec3<T> &vertex1,
                               const vec3<T> &vertex2)
      : v0(vertex0), v1(vertex1), v2(vertex2){};

  __host__ __device__ AABB<T> ComputeBBox() {
    return AABB<T>(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)),
                   min(v0.z, min(v1.z, v2.z)), max(v0.x, max(v1.x, v2.x)),
                   max(v0.y, max(v1.y, v2.y)), max(v0.z, max(v1.z, v2.z)));
  }
};

template <typename T> using TrianglePtr = Triangle<T> *;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Triangle<T> &x) {
  os << x.v0 << std::endl;
  os << x.v1 << std::endl;
  os << x.v2 << std::endl;
  return os;
}

template <typename T> __host__ __device__ T dot2(vec3<T> v) {
  return dot(v, v);
}

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p,
                                              TrianglePtr<T> tri_ptr) {
  vec3<T> a = tri_ptr->v0;
  vec3<T> b = tri_ptr->v1;
  vec3<T> c = tri_ptr->v2;

  vec3<T> ba = b - a;
  vec3<T> pa = p - a;
  vec3<T> cb = c - b;
  vec3<T> pb = p - b;
  vec3<T> ac = a - c;
  vec3<T> pc = p - c;
  vec3<T> nor = cross(ba, ac);

  return sqrt(
      (sign<T>(dot(cross(ba, nor), pa)) + sign<T>(dot(cross(cb, nor), pb)) +
           sign<T>(dot(cross(ac, nor), pc)) <
       2.0)
          ? min(min(dot2<T>(ba * clamp(dot(ba, pa) / dot2<T>(ba), 0.0, 1.0) -
                            pa),
                    dot2<T>(cb * clamp(dot(cb, pb) / dot2<T>(cb), 0.0, 1.0) -
                            pb)),
                dot2<T>(ac * clamp(dot(ac, pc) / dot2<T>(ac), 0.0, 1.0) - pc))
          : dot(nor, pa) * dot(nor, pa) / dot2<T>(nor));
}

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p, TrianglePtr<T> tri_ptr,
                                              vec3<T> *closest_point) {
  vec3<T> a = tri_ptr->v0;
  vec3<T> b = tri_ptr->v1;
  vec3<T> c = tri_ptr->v2;

  // vec3<T> ba = b - a;
  vec3<T> pa = p - a;
  // vec3<T> cb = c - b;
  vec3<T> pb = p - b;
  // vec3<T> ac = a - c;
  vec3<T> pc = p - c;
  // vec3<T> nor = cross(ba, ac);

  vec3<T> ab = b - a;
  vec3<T> ac = c - a;
  vec3<T> bc = c - b;
  // Compute parametric position s for projection P’ of P on AB,
  // P’ = A + s*AB, s = snom/(snom+sdenom)
  T snom = dot(p - a, ab);
  T sdenom = dot(p - b, a - b);
  // Compute parametric position t for projection P’ of P on AC,
  // P’ = A + t*AC, s = tnom/(tnom+tdenom)
  T tnom = dot(p - a, ac), tdenom = dot(p - c, a - c);
  if (snom <= static_cast<T>(0.0) && tnom <= static_cast<T>(0.0)) {
    *closest_point = a;
    return sqrt(dot(pa, pa));
  }
  // Compute parametric position u for projection P’ of P on BC,
  // P’ = B + u*BC, u = unom/(unom+udenom)
  T unom = dot(p - b, bc), udenom = dot(p - c, b - c);
  if (sdenom <= static_cast<T>(0.0) && unom <= static_cast<T>(0.0)) {
    *closest_point = b;
    return sqrt(dot(pb, pb));
  }
  if (tdenom <= static_cast<T>(0.0f) && udenom <= static_cast<T>(0.0f)) {
    *closest_point = c;
    return sqrt(dot(pc, pc));
  }
  // P is outside (or on) AB if the triple scalar product [N PA PB] <= 0
  vec3<T> n = cross(b - a, c - a);
  T vc = dot(n, cross(a - p, b - p));
  // If P outside AB and within feature region of AB,
  // return projection of P onto AB
  if (vc <= static_cast<T>(0.0f) && snom >= static_cast<T>(0.0f) &&
      sdenom >= static_cast<T>(0.0f)) {
    *closest_point = a + snom / (snom + sdenom) * ab;
    return sqrt(dot(p - *closest_point, p - *closest_point));
  }
  // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
  T va = dot(n, cross(b - p, c - p));
  // If P outside BC and within feature region of BC,
  // return projection of P onto BC
  if (va <= static_cast<T>(0.0f) && unom >= static_cast<T>(0.0f) &&
      udenom >= static_cast<T>(0.0f)) {
    *closest_point = b + unom / (unom + udenom) * bc;
    return sqrt(dot(p - *closest_point, p - *closest_point));
  }
  // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
  T vb = dot(n, cross(c - p, a - p));
  // If P outside CA and within feature region of CA,
  // return projection of P onto CA
  if (vb <= static_cast<T>(0.0f) && tnom >= static_cast<T>(0.0f) &&
      tdenom >= static_cast<T>(0.0f)) {
    *closest_point = a + tnom / (tnom + tdenom) * ac;
    return sqrt(dot(p - *closest_point, p - *closest_point));
  }
  // P must project inside face region. Compute Q using barycentric coordinates
  T u = va / (va + vb + vc);
  T v = vb / (va + vb + vc);
  T w = static_cast<T>(1.0f) - u - v; // = vc / (va + vb + vc))
  *closest_point = u * a + v * b + w * c;
  return sqrt(dot(p - *closest_point, p - *closest_point));
}

template <typename T>
__global__ void ComputeTriBoundingBoxes(Triangle<T> *triangles,
                                        int num_triangles, AABB<T> *bboxes) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    bboxes[idx] = triangles[idx].ComputeBBox();
  }
  return;
}

template <typename T> struct BVHNode {
public:
  AABB<T> bbox;

  TrianglePtr<T> tri_ptr;
  BVHNode<T> *left;
  BVHNode<T> *right;
  BVHNode<T> *parent;
  __host__ __device__ inline bool isLeaf() { return !left && !right; };

  // The index of the object contained in the node
  int idx;
};

template <typename T> using BVHNodePtr = BVHNode<T> *;

template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    bool
    checkOverlap(const AABB<T> &bbox1, const AABB<T> &bbox2) {
  return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) &&
         (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) &&
         (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

template <typename T, int QueueSize = 32>
__device__ T traverseBVH(const vec3<T> &queryPoint, BVHNodePtr<T> root,
                         long *closest_face, vec3<T> *closestPoint) {
  // Allocate traversal stack from thread-local memory,
  // and push NULL to indicate that there are no postponed nodes.
  PriorityQueue<T, BVHNodePtr<T>, QueueSize> queue;

  T root_dist = pointToAABBDistance(queryPoint, root->bbox);

  // int closest_index = -1;
  queue.insert_key(root_dist, root);

  BVHNodePtr<T> node = nullptr;

  T closest_distance = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
  // printf("%f %f %f\n", queryPoint.x, queryPoint.y, queryPoint.z);

  while (queue.get_size() > 0) {
    std::pair<T, BVHNodePtr<T>> output = queue.extract();
    T curr_distance = output.first;
    node = output.second;

    // Check each child node for overlap.
    BVHNodePtr<T> childL = node->left;
    BVHNodePtr<T> childR = node->right;

    T distance_left = pointToAABBDistance<T>(queryPoint, childL->bbox);
    T distance_right = pointToAABBDistance<T>(queryPoint, childR->bbox);

    // printf("%f, left = %f, right = %f\n", closest_distance, distance_left,
    // distance_right);

    if (distance_left <= closest_distance) {
      if (childL->isLeaf()) {
        // If  the child is a leaf then
        TrianglePtr<T> tri_ptr = childL->tri_ptr;
        vec3<T> curr_clos_point;
        T distance_left =
            pointToTriangleDistance<T>(queryPoint, tri_ptr, &curr_clos_point);
        if (distance_left <= closest_distance) {
          closest_distance = distance_left;
          *closest_face = childL->idx;
          *closestPoint = curr_clos_point;
        }
      } else {
        queue.insert_key(distance_left, childL);
      }
    }

    if (distance_right <= closest_distance) {
      if (childR->isLeaf()) {
        // If  the child is a leaf then
        TrianglePtr<T> tri_ptr = childR->tri_ptr;
        vec3<T> curr_clos_point;
        T distance_right =
            pointToTriangleDistance<T>(queryPoint, tri_ptr, &curr_clos_point);
        if (distance_right <= closest_distance) {
          closest_distance = distance_right;
          *closest_face = childR->idx;
          *closestPoint = curr_clos_point;
          // printf("Child R: %f, %d\n", closest_distance, *closest_face);
        }
      } else {
        queue.insert_key(distance_right, childR);
      }
    }
  }
  // printf("%d\n", *closest_face);

  return closest_distance;
}

template <typename T, int QueueSize = 32>
__global__ void findNearestNeighbor(vec3<T> *query_points, T *distances,
                                    vec3<T> *closest_points,
                                    long *closest_faces, BVHNodePtr<T> root,
                                    int num_points) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_points;
       idx += blockDim.x * gridDim.x) {
    vec3<T> query_point = query_points[idx];

    long closest_face;
    vec3<T> closest_point;

    T closest_distance =
        traverseBVH<T, QueueSize>(query_point, root, &closest_face, &closest_point);
    distances[idx] = closest_distance;
    closest_points[idx] = closest_point;
    closest_faces[idx] = closest_face;
  }
  return;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        expandBits(MortonCode v) {
  // Shift 16
  v = (v * 0x00010001u) & 0xFF0000FFu;
  // Shift 8
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  // Shift 4
  v = (v * 0x00000011u) & 0xC30C30C3u;
  // Shift 2
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        morton3D(T x, T y, T z) {
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  MortonCode xx = expandBits((MortonCode)x);
  MortonCode yy = expandBits((MortonCode)y);
  MortonCode zz = expandBits((MortonCode)z);
  return xx * 4 + yy * 2 + zz;
}

template <typename T>
__global__ void ComputeMortonCodes(Triangle<T> *triangles, int num_triangles,
                                   AABB<T> *scene_bb,
                                   MortonCode *morton_codes) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    // Fetch the current triangle
    Triangle<T> tri = triangles[idx];
    vec3<T> centroid = (tri.v0 + tri.v1 + tri.v2) / (T)3.0;

    T x = (centroid.x - scene_bb->min_t.x) /
          (scene_bb->max_t.x - scene_bb->min_t.x);
    T y = (centroid.y - scene_bb->min_t.y) /
          (scene_bb->max_t.y - scene_bb->min_t.y);
    T z = (centroid.z - scene_bb->min_t.z) /
          (scene_bb->max_t.z - scene_bb->min_t.z);

    morton_codes[idx] = morton3D<T>(x, y, z);
  }
  return;
}

__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    int
    LongestCommonPrefix(int i, int j, MortonCode *morton_codes,
                        int num_triangles, int *triangle_ids) {
  // This function will be called for i - 1, i, i + 1, so we might go beyond
  // the array limits
  if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
    return -1;

  MortonCode key1 = morton_codes[i];
  MortonCode key2 = morton_codes[j];

  if (key1 == key2) {
    // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
    // bits in key[1, 2]. Add the number of leading zeros between the
    // indices
    return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
  } else {
    // Keys are different
    return __clz(key1 ^ key2);
  }
}

template <typename T>
__global__ void BuildRadixTree(MortonCode *morton_codes, int num_triangles,
                               int *triangle_ids, BVHNodePtr<T> internal_nodes,
                               BVHNodePtr<T> leaf_nodes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles - 1;
       idx += blockDim.x * gridDim.x) {
    // if (idx >= num_triangles - 1)
    // return;

    int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
                                         num_triangles, triangle_ids);
    int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
                                         num_triangles, triangle_ids);
    // Find the direction of the range
    int direction = delta_next - delta_last >= 0 ? 1 : -1;

    int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
                                        num_triangles, triangle_ids);

    // Do binary search to compute the upper bound for the length of the range
    int lmax = 2;
    while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
                               num_triangles, triangle_ids) > delta_min) {
      lmax *= 2;
    }

    // Use binary search to find the other end.
    int l = 0;
    int divider = 2;
    for (int t = lmax / divider; t >= 1; divider *= 2) {
      if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
                              num_triangles, triangle_ids) > delta_min) {
        l = l + t;
      }
      t = lmax / divider;
    }
    int j = idx + l * direction;

    // Find the length of the longest common prefix for the current node
    int node_delta =
        LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
    int s = 0;
    divider = 2;
    // Search for the split position using binary search.
    for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
      if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
                              num_triangles, triangle_ids) > node_delta) {
        s = s + t;
      }
      t = (l + (divider - 1)) / divider;
    }
    // gamma in the Karras paper
    int split = idx + s * direction + min(direction, 0);

    // Assign the parent and the left, right children for the current node
    BVHNodePtr<T> curr_node = internal_nodes + idx;
    if (min(idx, j) == split) {
      curr_node->left = leaf_nodes + split;
      (leaf_nodes + split)->parent = curr_node;
    } else {
      curr_node->left = internal_nodes + split;
      (internal_nodes + split)->parent = curr_node;
    }
    if (max(idx, j) == split + 1) {
      curr_node->right = leaf_nodes + split + 1;
      (leaf_nodes + split + 1)->parent = curr_node;
    } else {
      curr_node->right = internal_nodes + split + 1;
      (internal_nodes + split + 1)->parent = curr_node;
    }
  }
  return;
}

template <typename T>
__global__ void CreateHierarchy(BVHNodePtr<T> internal_nodes,
                                BVHNodePtr<T> leaf_nodes, int num_triangles,
                                Triangle<T> *triangles, int *triangle_ids,
                                int *atomic_counters) {
  // int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // if (idx >= num_triangles)
  // return;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {

    BVHNodePtr<T> leaf = leaf_nodes + idx;
    // Assign the index to the primitive
    leaf->idx = triangle_ids[idx];

    Triangle<T> tri = triangles[triangle_ids[idx]];
    // Assign the bounding box of the triangle to the leaves
    leaf->bbox = tri.ComputeBBox();
    leaf->tri_ptr = &triangles[triangle_ids[idx]];

    BVHNodePtr<T> curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;

    // Increment the atomic counter
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    while (true) {
      // atomicAdd returns the old value at the specified address. Thus the
      // first thread to reach this point will immediately return
      if (curr_counter == 0)
        break;

      // Calculate the bounding box of the current node as the union of the
      // bounding boxes of its children.
      AABB<T> left_bb = curr_node->left->bbox;
      AABB<T> right_bb = curr_node->right->bbox;
      curr_node->bbox = left_bb + right_bb;
      // If we have reached the root break
      if (curr_node == internal_nodes)
        break;

      // Proceed to the parent of the node
      curr_node = curr_node->parent;
      // Calculate its position in the flat array
      current_idx = curr_node - internal_nodes;
      // Update the visitation counter
      curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    }
  }

  return;
}

template <typename T, int blockSize = NUM_THREADS>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T> *__restrict__ triangles,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size) {

#if PRINT_TIMINGS == 1
  // Create the CUDA events used to estimate the execution time of each
  // kernel.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  thrust::device_vector<AABB<T>> bounding_boxes(num_triangles);

  int gridSize = (num_triangles + blockSize - 1) / blockSize;
#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the bounding box for all the triangles
#if DEBUG_PRINT == 1
  std::cout << "Start computing triangle bounding boxes" << std::endl;
#endif
  ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, bounding_boxes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished computing triangle bounding_boxes" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Compute Triangle Bounding boxes = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the union of all the bounding boxes
  AABB<T> host_scene_bb = thrust::reduce(
      bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished Calculating scene Bounding Box" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Scene bounding box reduction = " << milliseconds << " (ms)"
            << std::endl;
#endif

  // TODO: Custom reduction ?
  // Copy the bounding box back to the GPU
  AABB<T> *scene_bb_ptr;
  cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
  cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
             cudaMemcpyHostToDevice);

  thrust::device_vector<MortonCode> morton_codes(num_triangles);
#if DEBUG_PRINT == 1
  std::cout << "Start Morton Code calculation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the morton codes for the centroids of all the primitives
  ComputeMortonCodes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, scene_bb_ptr, morton_codes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished calculating Morton Codes ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Morton code calculation = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if DEBUG_PRINT == 1
  std::cout << "Creating triangle ID sequence" << std::endl;
#endif
  // Construct an array of triangle ids.
  thrust::sequence(triangle_ids->begin(), triangle_ids->end());
#if DEBUG_PRINT == 1
  std::cout << "Finished creating triangle ID sequence ..." << std::endl;
#endif

  // Sort the triangles according to the morton code
#if DEBUG_PRINT == 1
  std::cout << "Starting Morton Code sorting!" << std::endl;
#endif

  try {
#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                        triangle_ids->begin());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
    std::cout << "Finished morton code sorting!" << std::endl;
#endif
#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Morton code sorting = " << milliseconds << " (ms)"
              << std::endl;
#endif
  } catch (thrust::system_error e) {
    std::cout << "Error inside sort: " << e.what() << std::endl;
  }

#if DEBUG_PRINT == 1
  std::cout << "Start building radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Construct the radix tree using the sorted morton code sequence
  BuildRadixTree<T><<<gridSize, blockSize>>>(
      morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
      internal_nodes, leaf_nodes);
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Building radix tree = " << milliseconds << " (ms)" << std::endl;
#endif
  // Create an array that contains the atomic counters for each node in the
  // tree
  thrust::device_vector<int> counters(num_triangles);

#if DEBUG_PRINT == 1
  std::cout << "Start Linear BVH generation" << std::endl;
#endif
  // Build the Bounding Volume Hierarchy in parallel from the leaves to the
  // root
  CreateHierarchy<T><<<gridSize, blockSize>>>(
      internal_nodes, leaf_nodes, num_triangles, triangles,
      triangle_ids->data().get(), counters.data().get());

  cudaCheckError();

#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
  std::cout << "Finished with LBVH generation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Hierarchy generation = " << milliseconds << " (ms)"
            << std::endl;
#endif

  cudaFree(scene_bb_ptr);
  return;
}

// void bvh_self_distance_queries_kernel(const torch::Tensor &triangles,
// torch::Tensor distances,
// torch::Tensor closest_points) {
// const auto batch_size = triangles.size(0);
// const auto num_triangles = triangles.size(1);

// thrust::device_vector<int> triangle_ids(num_triangles);

// int blockSize = NUM_THREADS;
// int gridSize = (num_triangles + blockSize - 1) / blockSize;
// }

void bvh_distance_queries_kernel(const torch::Tensor &triangles,
                                 const torch::Tensor &points,
                                 torch::Tensor *distances,
                                 torch::Tensor *closest_points,
                                 torch::Tensor *closest_faces,
                                 int queue_size=128) {
  const auto batch_size = triangles.size(0);
  const auto num_triangles = triangles.size(1);
  const auto num_points = points.size(1);

  thrust::device_vector<int> triangle_ids(num_triangles);

  int blockSize = NUM_THREADS;
  int gridSize = (num_triangles + blockSize - 1) / blockSize;

#if PRINT_TIMINGS == 1
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  // Construct the bvh tree
  // AT_DISPATCH_FLOATING_TYPES(
  // triangles.type(), "bvh_tree_building", ([&] {
  thrust::device_vector<BVHNode<float>> leaf_nodes(num_triangles);
  thrust::device_vector<BVHNode<float>> internal_nodes(num_triangles - 1);
  auto triangle_float_ptr = triangles.data<float>();

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  for (int bidx = 0; bidx < batch_size; ++bidx) {

    Triangle<float> *triangles_ptr =
        (TrianglePtr<float>)triangle_float_ptr + num_triangles * bidx;

    // thrust::fill(collisionIndices.begin(), collisionIndices.end(),
    // make_long2(-1, -1));

#if DEBUG_PRINT == 1
    std::cout << "Start building BVH" << std::endl;
#endif
    buildBVH<float, NUM_THREADS>(
            internal_nodes.data().get(), leaf_nodes.data().get(),
            triangles_ptr, &triangle_ids, num_triangles, batch_size);
#if DEBUG_PRINT == 1
    std::cout << "Successfully built BVH" << std::endl;
#endif

#if DEBUG_PRINT == 1
    std::cout << "Start BVH traversal" << std::endl;
#endif
    auto distances_float_ptr = distances->data<float>();
    auto closest_points_float_ptr = closest_points->data<float>();
    auto closest_faces_long_ptr = closest_faces->data<long>();

    vec3<float> *closest_points_ptr =
        (vec3<float> *)closest_points_float_ptr + num_points * bidx;
    long *closest_faces_ptr =
        (long *)closest_faces_long_ptr + num_points * bidx;
    float *distances_ptr = (float *)distances_float_ptr + num_points * bidx;

    vec3<float> *points_ptr =
        (vec3<float> *)points.data<float>() + num_points * bidx;

    if (queue_size == 32) {
        findNearestNeighbor<float, 32><<<32 * numSMs, 256>>>(
                points_ptr, distances_ptr, closest_points_ptr, closest_faces_ptr,
                internal_nodes.data().get(), num_points);
    }
    else if (queue_size == 64){
        findNearestNeighbor<float, 64><<<32 * numSMs, 256>>>(
                points_ptr, distances_ptr, closest_points_ptr, closest_faces_ptr,
                internal_nodes.data().get(), num_points);
    }
    else if (queue_size == 128){
        findNearestNeighbor<float, 128><<<32 * numSMs, 256>>>(
                points_ptr, distances_ptr, closest_points_ptr, closest_faces_ptr,
                internal_nodes.data().get(), num_points);
    }
    else if (queue_size == 256){
        findNearestNeighbor<float, 256><<<32 * numSMs, 256>>>(
                points_ptr, distances_ptr, closest_points_ptr, closest_faces_ptr,
                internal_nodes.data().get(), num_points);
    }

    cudaCheckError();

#if DEBUG_PRINT == 1
    std::cout << "Successfully finished BVH traversal" << std::endl;
#endif
  }
  // }));
}
