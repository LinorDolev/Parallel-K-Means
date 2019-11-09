#include "Cluster.h"

void initClustersWithFirstKPoints(int k, Cluster* clusters, Point* kPoints)
{
	int i;

	for (i = 0; i < k; i++)
	{
		clusters[i].id = i;
		clusters[i].logicalSize = 0;
		clusters[i].physicalSize = 1;
		clusters[i].points = (Point*)calloc(clusters[i].physicalSize, sizeof(Point));
		checkAllocation(clusters[i].points);
		clusters[i].coor = kPoints[i].coor;
	}

}

void resetClusters(int k, Cluster* clusters, Point* kPoints)
{
	int i;

	for (i = 0; i < k; i++)
	{
		clusters[i].logicalSize = 0;
		clusters[i].coor = kPoints[i].coor;
	}
}

void addPoint(Cluster* cluster, Point* point)
{
	if (cluster->logicalSize == cluster->physicalSize)
	{
		cluster->physicalSize *= 2;
		cluster->points = (Point*)realloc(cluster->points, sizeof(Point)*cluster->physicalSize);
		checkAllocation(cluster->points);
	}
	point->indexInCluster = cluster->logicalSize;
	cluster->points[cluster->logicalSize++] = *point;
	point->clusterId = cluster->id;
}

void removePoint(Cluster* cluster, Point* pointToRemove, Point* points)
{
	int removeIndex = pointToRemove->indexInCluster;

	if (pointToRemove->clusterId != cluster->id)
		return;

	cluster->points[removeIndex] = cluster->points[cluster->logicalSize - 1];
	cluster->logicalSize--;
	cluster->points[removeIndex].indexInCluster = removeIndex;

	//updating the index of original point in points array
	points[cluster->points[removeIndex].id].indexInCluster = removeIndex;


	pointToRemove->indexInCluster = NO_CLUSTER;
	pointToRemove->clusterId = NO_CLUSTER;

}

void calculateDiameter(Cluster* cluster)
{
	int i, j, numberOfPoints = cluster->logicalSize;
	double maxDiameter = 0, dist;
	//printf("\ninside calculate diameter:\nall dist of cluster %d:\n",cluster->id);
	for (i = 0; i < numberOfPoints; i++)
	{
		for (j = i + 1; j < numberOfPoints; j++)
		{
			dist = distance(&cluster->points[i].coor, &cluster->points[j].coor);
			//printf("\ndist = %lf", dist);
			if (dist > maxDiameter)
			{
				maxDiameter = dist;
			}
		}
	}
	//printf("\ndiameter befor update = %lf", cluster->diameter);
	//printf("\ninside calculateDiameter:\nCluster:%d\nmaxDiameter= %lf\n",cluster->id, maxDiameter);

	cluster->diameter = maxDiameter;
}