#pragma once

#ifndef __CLUSTER_H
#define __CLUSTER_H

#include <stdlib.h>
#include <stdio.h>

#include "Point.h"

typedef struct cluster_t
{
	int id;
	Coordinate coor = { 0,0 }; //x,y
	Point* points;
	int physicalSize, logicalSize;
	double diameter;
}Cluster;

#endif

void initClustersWithFirstKPoints(int k, Cluster* clusters, Point* kPoints);
void resetClusters(int k, Cluster* clusters, Point* kPoints);

void addPoint(Cluster* cluster, Point* point);
void removePoint(Cluster* cluster, Point* pointToRemove, Point* points);
void calculateDiameter(Cluster* cluster);