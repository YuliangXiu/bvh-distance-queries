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

#endif // ifndef DEFINITIONS_H
