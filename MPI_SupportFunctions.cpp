#include "MPI_SupportFunctions.h"

void createMPI_CoorType(MPI_Datatype* MPI_CoorType) {
	MPI_Datatype type[] = { MPI_DOUBLE, MPI_DOUBLE };

	int blocklen[] = { 1,1 };

	MPI_Aint disp[2];
	
	disp[0] = offsetof(Coordinate, x);
	disp[1] = offsetof(Coordinate, y);

	MPI_Type_create_struct(2, blocklen, disp, type, MPI_CoorType);
	MPI_Type_commit(MPI_CoorType);
}

void createMPI_PointType(MPI_Datatype* MPI_PointType, MPI_Datatype MPI_CoorType) 
{
	MPI_Datatype type[] = { MPI_CoorType, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT};

	int blocklen[] = { 1, 1, 1, 1, 1, 1 };

	MPI_Aint disp[6];

	disp[0] = offsetof(Point, coor);
	disp[1] = offsetof(Point, vx);
	disp[2] = offsetof(Point, vy);
	disp[3] = offsetof(Point, clusterId);
	disp[4] = offsetof(Point, id);
	disp[5] = offsetof(Point, indexInCluster);

	MPI_Type_create_struct(6, blocklen, disp, type, MPI_PointType);
	MPI_Type_commit(MPI_PointType);
}

void createMPI_InfoType(MPI_Datatype* MPI_InfoType)
{
	MPI_Datatype type[] = { MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	 
	int blocklen[] = { 1, 1, 1, 1, 1, 1 };

	MPI_Aint disp[6];

	disp[0] = offsetof(Info, N);
	disp[1] = offsetof(Info, K);
	disp[2] = offsetof(Info, LIMIT);
	disp[3] = offsetof(Info, QM);
	disp[4] = offsetof(Info, T);
	disp[5] = offsetof(Info, dT);

	MPI_Type_create_struct(6, blocklen, disp, type, MPI_InfoType);
	MPI_Type_commit(MPI_InfoType);

}

Point* broadcastInitialData(Info* info, Point* points, int rank) 
{
	MPI_Datatype MPI_CoorType, MPI_PointType, MPI_InfoType;

	createMPI_CoorType(&MPI_CoorType);
	createMPI_InfoType(&MPI_InfoType);
	createMPI_PointType(&MPI_PointType, MPI_CoorType);

	MPI_Bcast(info, 1, MPI_InfoType,ROOT, MPI_COMM_WORLD);

	if (rank != ROOT)
		points = (Point*)calloc(info->N, sizeof(Point));

	MPI_Bcast(points, info->N, MPI_PointType, ROOT, MPI_COMM_WORLD);

	return points;
}

Coordinate* getAllClustersCoordinates(Cluster* clusters, int k)
{
	int i;
	Coordinate* coordinates = (Coordinate*)calloc(k, sizeof(Coordinate));

	for (i = 0; i < k; i++)
	{
		coordinates[i] = clusters[i].coor;
	}

	return coordinates;
}

void setAllClustersCoordinates(Cluster* clusters, Coordinate* coordinates, int k)
{
	int i;

	for (i = 0; i < k; i++)
	{
		clusters[i].coor = coordinates[i];
	}
}

int checkIfAnyProcessReachedQuality(Info* info, Cluster* clusters, MPI_Datatype MPI_CoorType,
	int stop, int numprocs, int rank, double* quality, double* time) //assumes stop = currQM < QM || t > T
{
	int i, toStop = 0;
	int* recvStops = (int*)calloc(numprocs, sizeof(int));
	checkAllocation(recvStops);

	double data[] = { *quality, *time };

	MPI_Allgather(&stop, 1, MPI_INT, recvStops, 1, MPI_INT, MPI_COMM_WORLD);
	
	for (i = 0; i < numprocs; i++)
	{
		if (recvStops[i])
		{
			if (rank == i && rank != ROOT) //This Process reached the desired quality and sends ROOT his centers
			{
				Coordinate* myClustersCoordinates = getAllClustersCoordinates(clusters, info->K);
				MPI_Send(myClustersCoordinates, info->K, MPI_CoorType, ROOT, 0, MPI_COMM_WORLD);
				MPI_Send(data, 2, MPI_DOUBLE, ROOT,0, MPI_COMM_WORLD);
				free(myClustersCoordinates);
			}
			else if (rank != i && rank == ROOT) 
			{
				MPI_Status status;
				Coordinate* clustersCoordinates = (Coordinate*)calloc(info->K, sizeof(Coordinate));
				
				MPI_Recv(clustersCoordinates, info->K, MPI_CoorType, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(data, 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

				setAllClustersCoordinates(clusters, clustersCoordinates, info->K);
				
				*quality = data[0];
				*time = data[1];

				free(clustersCoordinates);
			}

			toStop = 1;
			break;
		}
	}

	free(recvStops);
	return toStop;
}