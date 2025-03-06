#pragma once
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MIN_COUNT_ARGS 3
#define EPSILON 0.0001

typedef struct {
    double** A;
    double* u;
    double* b;
    double* x;
    double* residualVector;
} MatrixSet;

void solveBySimpleIteration(char** argv);

double** allocateMatrix(int N);
void initializeMatrix(double** A, int N);
void deallocateMatrix(double** A, int N);
void initializeBRandomVector(double** A, double* b, double* u, int N);
void initializeSolutionVector(double* u, int N);
void initializeZeroVector(double* x, int N);
void printMatrix(double** A, int N);
void printVector(double* vect, int N);
void checkResult(double* A, double* B, int N);

double computeRelativeResidualNorm(double* residualVector, double** A, double* x, double* b, int N);
void matrixMult(double* residualVector, double** A, double* x, int N);
void vectorSub(double* residualVector, double* b, int N);
void scalarMult(double* residualVector, double scalar, int N);
void calculateNewIteration(double* residualVector, double** A, double* x, double* b, int N);
void calculateNewIterationv(double* residualVector, double** A, double* x, double* b, int N, double tay, int i);

void matricesFree(MatrixSet matrices, int N);

double calculateTAY(double** A, int N);