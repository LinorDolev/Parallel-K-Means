
#include "KMeans.h"

void kMeans(Point* points, Cluster* clusters, Info* info, int* iterNum)
{
	
	int hasChanged = 0, i;
	*iterNum = 0;
	resetClusters(info->K, clusters, points);
	//initClustersWithFirstKPoints(info->K, clusters, points);

	do
	{
		hasChanged = matchPointsToClusters(info->K, clusters, info->N, points);
		
		calculateNewCenters(info->K, clusters);
		
	} while ((*iterNum)++ < info->LIMIT && hasChanged);
}

int findClosestClusterId(Point* point, Cluster* clusters, int k)
{
	int minIndex = 0, j;
	double minDist = INT_MAX;

	for (j = 0; j < k; j++)
	{
		double dist = distance(&point->coor, &clusters[j].coor);
		//printf("\nInside findClosestClusterId all dist: \ndist= %lf\n", dist);
		if (dist < minDist)
		{
			minDist = dist;
			minIndex = j;
		}
	}
	//printf("\nFinalMinDist= %lf\nMinIndexCluster= %d\n", minDist, minIndex);
	return minIndex;
}

int matchPointsToClusters(int k, Cluster* clusters, int numOfPoints, Point* points)
{
	int i, j, minIndex, hasChanged = 0;
	double minDist;

	for (i = 0; i < numOfPoints; i++)
	{

		minIndex = findClosestClusterId(&points[i], clusters, k);

		if (minIndex != points[i].clusterId)
		{
			hasChanged = 1;
			//printf("\ncluster id of point befor remove= %d\n", points[i].clusterId);
			removePoint(&clusters[points[i].clusterId], &points[i], points);
			//printf("\ncluster id of point after remove= %d\n", points[i].clusterId);
			addPoint(&clusters[minIndex], &points[i]);
			//printf("\ncluster id of point after add point= %d\n", points[i].clusterId);
		}

	}

	return hasChanged;
}

void calculateNewCenters(int k, Cluster* clusters)
{
	int i, j, numOfPoints;
	double newCenterX, newCenterY;

	for (i = 0; i < k; i++)
	{
		numOfPoints = clusters[i].logicalSize;
		newCenterX = 0;
		newCenterY = 0;

		for (j = 0; j < numOfPoints; j++)
		{
			newCenterX += clusters[i].points[j].coor.x;
			newCenterY += clusters[i].points[j].coor.y;
		}

		clusters[i].coor.x = newCenterX / clusters[i].logicalSize;
		clusters[i].coor.y = newCenterY / clusters[i].logicalSize;
	}
}