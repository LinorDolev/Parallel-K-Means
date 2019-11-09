#ifndef __KERNEL_H
#define __KERNEL_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.h"

__global__ void movePointsInTimeKernel(Point *points, int N, double parallelDt);

void increaseTimeAndCleanPointsWithCUDA(Point* cudaPoints, Point* cpuPoints, Info* info, int offset);

Point* cudaPointsAlloc(Info* info, Point* points);

void checkAndHandleErrors(cudaError_t cudaStatus, const char* errorMessage, Point* cudaPoints);

#endif