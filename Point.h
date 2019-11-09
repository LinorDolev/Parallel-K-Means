#pragma once

#ifndef __POINT_H
#define __POINT_H

#define NO_CLUSTER -1

typedef struct coor_t {
	double x;
	double y;
}Coordinate;

typedef struct point_t {
	Coordinate coor;
	double vx;
	double vy;
	int clusterId;
	int id;
	int indexInCluster = NO_CLUSTER; //index in cluster
}Point;

typedef struct info_t {
	int N;
	int K;
	int LIMIT;
	double QM;
	double T;
	double dT;
}Info;


void initPoint(Point* point, double x, double y, double vx, double vy, int clusterId, int id);

Point* readDataSet(const char* fileName, Info* info);

double distance(const Coordinate* p1, const Coordinate* p2);

void printPoint(Point* point);

void checkAllocation(void* ptr);

#endif