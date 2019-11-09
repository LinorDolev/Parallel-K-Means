#pragma once

#ifndef __KMEANS_H
#define __KMEANS_H

#include <stdlib.h>
#include <stdio.h>

#include "Cluster.h"

void kMeans(Point* points, Cluster* clusters, Info* info, int* iterNum);

int findClosestClusterId(Point* point, Cluster* clusters, int k);

int matchPointsToClusters(int k, Cluster* clusters, int numOfPoints, Point* points);

void calculateNewCenters(int k, Cluster* clusters);

#endif