#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "point.h"

void initPoint(Point* point, double x, double y, double vx, double vy, int clusterId, int id)
{
	point->coor.x = x;
	point->coor.y = y;
	point->vx = vx;
	point->vy = vy;
	point->clusterId = clusterId;
	point->id = id;
}

Point* readDataSet(const char* fileName, Info* info)
{
	int i;
	double x, y, vx, vy;

	FILE* file = fopen(fileName, "r");
	Point* points;

	if (!file)
	{
		printf("Error reading from file, make sure the INPUT_PATH is correct!\n");
		exit(1);
	}

	fscanf(file, "%d %d %lf %lf %d %lf", &(info->N), &(info->K), &(info->T), &(info->dT), &(info->LIMIT), &(info->QM));

	points = (Point*)calloc(info->N, sizeof(Point));
	checkAllocation(points);

	for (i = 0; i < info->N; i++)
	{
		fscanf(file, "%lf %lf %lf %lf", &x, &y, &vx, &vy);
		initPoint(&points[i], x, y, vx, vy, NO_CLUSTER, i);
		//printPoint(&points[i]);
	}

	fclose(file);
	return points;
}

double distance(const Coordinate* p1, const Coordinate* p2)
{
	double deltaX = p1->x - p2->x;
	double deltaY = p1->y - p2->y;

	return sqrt((deltaX * deltaX) + (deltaY * deltaY));
}

void printPoint(Point* point)
{
	printf("\n%lf %lf %lf %lf", point->coor.x, point->coor.y, point->vx, point->vy);
}

void checkAllocation(void* ptr)
{
	if (!ptr)
	{
		printf("\n\n***** OUT OF MEMORY! *****\n\n");
		exit(2);
	}
}