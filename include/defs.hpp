#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

template <typename T>
using vec3 = typename std::conditional<std::is_same<T, float>::value, float3,
                                       double3>::type;

template <typename T>
using vec2 = typename std::conditional<std::is_same<T, float>::value, float2,
                                       double2>::type;

float3 make_float3(double3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

float3 make_float3(double x, double y, double z) {
    return make_float3(x, y, z);
}

double3 make_double3(float3 vec) {
    return make_double3(vec.x, vec.y, vec.z);
}

double3 make_double3(float x, float y, float z) {
    return make_double3(x, y, z);
}

template <typename T>
__host__ __device__
vec3<T> make_vec3(T x, T y, T z) {
}

template <>
__host__ __device__
vec3<float> make_vec3(float x, float y, float z) {
    return make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

template <>
__host__ __device__
vec3<double> make_vec3(double x, double y, double z) {
    return make_double3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}

#endif // ifndef DEFINITIONS_H
