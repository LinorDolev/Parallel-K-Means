
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "KMeans.h"
#include "MPI_SupportFunctions.h"
#include "Kernel.h"

Cluster* linor(Point* points, Info* info, double* quality, double* t, int* iterNum, int rank, MPI_Datatype MPI_CoorType);

double qualityMeasure(Cluster* clusters, int k);

void increaseTimeAndCleanPoints(Point* points, Info* info, int offset);

void printResultToFile(const char* fileName, Cluster* clusters, double quality, double t, int k);


void main(int argc, char* argv[])
{
	const char* INPUT_PATH = "C:\\Users\\Linor\\Downloads\\data\\data\\data.txt";//"C:\\Users\\Linor\\Desktop\\input.txt";
	const char* OUTPUT_PATH = "C:\\Users\\Linor\\Desktop\\output.txt";

	int iterNum = 0, rank = 0;
	double quality = 0, t = 0;

	Cluster* clusters = NULL;
	Point* points;
	Info info;
	
	MPI_Datatype MPI_CoorType = NULL, MPI_PointType = NULL, MPI_InfoType = NULL;
	
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (rank == ROOT)
	{
		points = readDataSet(INPUT_PATH, &info);
		printf("%d %d %lf %lf %d %lf", (info.N), (info.K), (info.T), (info.dT), (info.LIMIT), (info.QM));
	}

	points = broadcastInitialData(&info, points, rank);
		
	clusters = linor(points, &info, &quality, &t, &iterNum, rank, MPI_CoorType);
	
	//printf("\nAfter linor:\nquality= %lf\ntime= %lf\niterNum= %d\n", quality, t, iterNum);
	
	if(rank == ROOT)
		printResultToFile(OUTPUT_PATH, clusters, quality, t, info.K);


	MPI_Finalize();
	//Free all dynamic alocated memory
	free(points);
	free(clusters);
}

Cluster* linor(Point* points, Info* info, double* quality, double* t, int* iterNum, int rank, MPI_Datatype MPI_CoorType)
{
	int i, stop = 0, numprocs;
	double currentQM = info->QM + 1;
	
	Cluster* clusters = (Cluster*)calloc(info->K, sizeof(Cluster));
	checkAllocation(clusters);

//	Point* _cudaPoints = cudaPointsAlloc(info, points);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	initClustersWithFirstKPoints(info->K, clusters, points);

	*t = rank * info->dT;
	//increaseTimeAndCleanPoints(points, info, rank);
	//increaseTimeAndCleanPointsWithCUDA(_cudaPoints, points, info, rank);
	do
	{
		kMeans(points, clusters, info, iterNum);
	
		currentQM = qualityMeasure(clusters, info->K);
		//printf("\nAfter qualityMeasure\ncurrentQM= %lf\nQM= %lf\ntime= %lf\n", currentQM, info->QM, *t);

		stop = (currentQM <= info->QM) || *t >= info->T;
		
		stop = checkIfAnyProcessReachedQuality(info, clusters, MPI_CoorType, stop, numprocs, rank, &currentQM, t);

		if (stop)
		{
			if ((*t) < info->T)
			{
				(*t) += info->dT * numprocs;
				//printf("\nCurrent Time= %lf\n", *t);
				
			//	increaseTimeAndCleanPointsWithCUDA(_cudaPoints, points, info, numprocs);
			}
		}
	
	} while (!stop);

	*quality = currentQM;

//	cudaFree(_cudaPoints);

	return clusters;
}

double qualityMeasure(Cluster* clusters, int k)
{
	int i, j;
	double quality = 0;

#pragma omp parallel for
	for (i = 0; i < k; i++)
	{
		calculateDiameter(&clusters[i]);
	}

	for (i = 0; i < k; i++)
	{
		for (j = i + 1; j < k; j++)
		{
			quality += (clusters[i].diameter + clusters[j].diameter) / distance(&clusters[i].coor, &clusters[j].coor);
		}
	}
	return (quality) / (k * (k - 1));
}

/* DELETE UNECESSARY FUNCTION - BEFORE CUDA*/
void increaseTimeAndCleanPoints(Point* points, Info* info, int offset)
{
	int i, numOfPoints = info->N;
	double parallelDt = info->dT * offset;

	for (i = 0; i < numOfPoints; i++)
	{
		points[i].coor.x += points[i].vx * parallelDt;
		points[i].coor.y += points[i].vy * parallelDt;

		points[i].clusterId = points[i].indexInCluster = NO_CLUSTER;
	}
}

void printResultToFile(const char* fileName, Cluster* clusters, double quality, double t, int k)
{
	int i, numOfPoints = 0;
	FILE* file = fopen(fileName, "w");

	if (!file)
	{
		printf("Error writing to file, make sure that the OUTPUT_PATH in Main.cpp is correct!\n");
		exit(3);
	}

	fprintf(file, "First occurrence at t = %lf with q = %lf \nCenters of the clusters: \n\n", t, quality);

	for (i = 0; i < k; i++)
	{
		fprintf(file, "%3.3lf %3.3lf\n", clusters[i].coor.x, clusters[i].coor.y);
		printf("\n\nid: %d (%3.3lf, %3.3lf) num of points: %d diameter: %lf\n", clusters[i].id, clusters[i].coor.x, clusters[i].coor.y, clusters[i].logicalSize, clusters[i].diameter);
		numOfPoints += clusters[i].logicalSize;
	}
	printf("\nTotal number of points: %d\n", numOfPoints);
	fclose(file);
}