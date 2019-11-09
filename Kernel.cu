
#include "Kernel.h"

__global__ void movePointsInTimeKernel(Point *points, int N, double parallelDt)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (index >= N)
		return;

	points[index].coor.x += points[index].vx * parallelDt;
	points[index].coor.y += points[index].vy * parallelDt;
	points[index].clusterId = points[index].indexInCluster = NO_CLUSTER;
}

void increaseTimeAndCleanPointsWithCUDA(Point* cudaPoints, Point* cpuPoints, Info* info, int offset)
{
	cudaDeviceProp prop;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&prop, 0);

	int numOfBlocks = info->N / prop.maxThreadsPerBlock;
	int numOfThreadsRemain = info->N % prop.maxThreadsPerBlock;
	int numOfThreads = numOfBlocks == 0 ? numOfThreadsRemain : prop.maxThreadsPerBlock;
	double parallelDt = info->dT * offset;

	checkAndHandleErrors(cudaStatus, "Error in cudaGetDeviceProps", cudaPoints);
	//movePointsInTimeKernel <<< numOfBlocks + 1, numOfThreads > > >(points, info->N, parallelDt);
	
	cudaStatus = cudaDeviceSynchronize();
	checkAndHandleErrors(cudaStatus, "Error in DeviceSynchronized", cudaPoints);

	cudaStatus = cudaMemcpy(cpuPoints, cudaPoints, sizeof(Point)* info->N, cudaMemcpyDeviceToHost);
	checkAndHandleErrors(cudaStatus, "Error in Memcpy results", cudaPoints);

}



Point* cudaPointsAlloc(Info* info, Point* points)
{
	cudaError_t cudaStatus;
	Point* cudaPoints;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkAndHandleErrors(cudaStatus, "Error in setDevice", NULL);

	cudaStatus = cudaMalloc(&cudaPoints, sizeof(Point) * info->N);
	checkAndHandleErrors(cudaStatus, "Error in cudaMalloc", cudaPoints);

	cudaStatus = cudaMemcpy(cudaPoints, points, info->N * sizeof(Point), cudaMemcpyHostToDevice);
	checkAndHandleErrors(cudaStatus, "Error in cudaMemcpy", cudaPoints);

	return cudaPoints;
}



void checkAndHandleErrors(cudaError_t cudaStatus, const char* errorMessage, Point* cudaPoints)
{
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(cudaPoints);
	}
}
