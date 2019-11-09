
#ifndef __MPI_SUPPORT_FUNCTIONS
#define __MPI_SUPPORT_FUNCTIONS

#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>

#include "Cluster.h"

#define ROOT 0

void createMPI_CoorType(MPI_Datatype* MPI_CoorType);

void createMPI_PointType(MPI_Datatype* MPI_PointType, MPI_Datatype MPI_CoorType);

void createMPI_InfoType(MPI_Datatype* MPI_InfoType);

Point* broadcastInitialData(Info* info, Point* points, int rank);

Coordinate* getAllClustersCoordinates(Cluster* clusters, int k);

void setAllClustersCoordinates(Cluster* clusters, Coordinate* coordinates, int k);

int checkIfAnyProcessReachedQuality(Info* info, Cluster* clusters, MPI_Datatype MPI_CoorType,
	int stop, int numprocs, int rank, double* quality, double* time);




#endif