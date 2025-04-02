#pragma once
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MIN_COUNT_ARGS 2
#define EPSILON 0.01

typedef struct {
    double* A;
    double* u;
    double* b;
    double* bRoot;
    double* x;
    double* xNew;
    double* residualVector;
} MatrixSet;

typedef struct {
    int blockSize;
    int extra;
    int startRow;
    int endRow;
    int localRows; 
} DataThreads;

void initializeDataThreads(DataThreads* dataThread, int N, int rank, int size);
void memoryAllocation(MatrixSet* matrices, DataThreads* dataThreads, int N, int rank);
void matricesFree(MatrixSet* matrices);

void solveBySimpleIteration(char** argv);

void initializeMatrix(double* A, int N, DataThreads* dataThread);
void initializeBRandomVector(double* A, double* b, double* bRoot, double* u, int N, DataThreads* dataThread);
void initializeSolutionVector(double* u, int N);
void initializeZeroVector(double* x, int N);

//Вывод матриц
void printMatrix(double* A, int SS, int N, int rank);
void printVector(double* vect, int N, int rank);

//Проверка результата
void checkResult(double* A, double* B, int N);

// Основные вычисления
double computeRelativeResidualNorm(double* residualVector, double* A, double* x, double* b, int N, DataThreads* dataThread);
void matrixMult(double* residualVector, double* A, double* x, int N, DataThreads* dataThread);
void vectorSub(double* residualVector, double* b, int N, DataThreads* dataThread);
void scalarMult(double* residualVector, double scalar, int N, DataThreads* dataThread);
void calculateNewIteration(double* residualVector, double* A, double* x, double* xNew, double* b, double tay, int N, DataThreads* dataThread);
double calculateTAY(double* A, int N, DataThreads* dataThread);