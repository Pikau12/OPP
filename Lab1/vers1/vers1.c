#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define _USE_MATH_DEFINES
#define MIN_COUNT_ARGS 2
#define EPSILON 0.001

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

void initializeDataThread(DataThreads* dataThread, int N, int size, int rank) {
    dataThread->blockSize = N / size;
    dataThread->extra = N % size;
    dataThread->startRow = rank * dataThread->blockSize + (rank < dataThread->extra ? rank : dataThread->extra);
    dataThread->endRow = dataThread->startRow + dataThread->blockSize + (rank < dataThread->extra ? 1 : 0);
    dataThread->localRows = dataThread->endRow - dataThread->startRow;
}

void memoryAllocation(MatrixSet* matrices, int N, DataThreads* dataThread) {
    matrices->A = (double*)malloc(sizeof(double) * dataThread->localRows * N);
    matrices->u = (double*)malloc(sizeof(double) * N);
    matrices->b = (double*)malloc(sizeof(double) * N);
    matrices->bRoot = (double*)malloc(sizeof(double) * N);
    matrices->x = (double*)malloc(sizeof(double) * N);
    matrices->xNew = (double*)malloc(sizeof(double) * N);
    matrices->residualVector = (double*)malloc(sizeof(double) * dataThread->localRows);

    if (!matrices->A || !matrices->u || !matrices->b || 
        !matrices->bRoot || !matrices->x || !matrices->xNew || !matrices->residualVector) {
        fprintf(stderr, "Memory allocation failed in process\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}


void matricesFree(MatrixSet* matrices) {
    free(matrices->A);
    free(matrices->b);
    free(matrices->bRoot);
    free(matrices->x);
    free(matrices->xNew);
    free(matrices->u);
    free(matrices->residualVector);

    matrices->A = NULL;
    matrices->b = NULL;
    matrices->bRoot = NULL;
    matrices->x = NULL;
    matrices->xNew = NULL;
    matrices->u = NULL;
    matrices->residualVector = NULL;
}

void initializeMatrix(double* matrix, int N, DataThreads* dataThread) {
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        int global_i = dataThread->startRow + local_i;
        for (int j = 0; j < N; j++) {
            matrix[local_i * N + j] = (j == global_i) ? 2.0 : 1.0;
        }
    }
}

void initializeBRandomVector(double* A, double* b, double* bRoot, double* u, int N, DataThreads* dataThread) {
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        b[local_i] = 0.0;
        for (int j = 0; j < N; j++) {
            b[local_i] += A[local_i * N + j] * u[j];
        }
    }

    int recvcounts[dataThread->blockSize];
    int displs[dataThread->blockSize];
    for (int i = 0; i < dataThread->blockSize; i++) {
        recvcounts[i] = dataThread->blockSize + (i < dataThread->extra ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
    }

    MPI_Allgatherv(b, dataThread->localRows, MPI_DOUBLE,bRoot, recvcounts, displs, MPI_DOUBLE,MPI_COMM_WORLD);
    memcpy(b, bRoot, N * sizeof(double));
}

void initializeZeroVector(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void initializeSolutionVector(double* u, int N) {
    for (int i = 0; i < N; i++) {
        u[i] = sin(2 * M_PI * i / N);        
    }
}

void matrixMult(double* result, double* A, double* x, int N, DataThreads* dataThread) {
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        result[local_i] = 0.0;
        for (int j = 0; j < N; j++) {
            result[local_i] += A[local_i * N + j] * x[j];
        }
    }
}

double calculateTAY(double* A, int N, DataThreads* dataThread) {
    double localMaxSum = 0.0;
    for (int i = 0; i < dataThread->blockSize; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < N; j++) {
            rowSum += fabs(A[i * N + j]);
        }
        if (rowSum > localMaxSum) {
            localMaxSum = rowSum;
        }
    }
    
    double globalMaxSum;
    MPI_Allreduce(&localMaxSum, &globalMaxSum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return 1.0 / globalMaxSum;
}

double computeRelativeResidualNorm(double* residualVector, double* A, double* x, double* b, int N, DataThreads* dataThread) {
    memset(residualVector, 0, sizeof(double) * dataThread->localRows);
    matrixMult(residualVector, A, x, N, dataThread);
    
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        residualVector[local_i] -= b[dataThread->startRow + local_i];
    }
    
    double local_r = 0.0, local_b = 0.0;
    for (int i = 0; i < dataThread->localRows; i++) {
        local_r += residualVector[i] * residualVector[i];
        local_b += b[dataThread->startRow + i] * b[dataThread->startRow + i];
    }
   
    double global_r, global_b;
    MPI_Allreduce(&local_r, &global_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_b, &global_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return sqrt(global_r/global_b);
}

void calculateNewIteration(double* residualVector, double* A, double* x, double* xNew, double* b, double tay, int N, DataThreads* dataThread) {
    memset(residualVector, 0, sizeof(double) * dataThread->localRows);
    matrixMult(residualVector, A, x, N, dataThread);
    for (int i = 0; i < dataThread->localRows; i++) {
        residualVector[i] -= b[dataThread->startRow + i];
    }

    for (int i = 0; i < dataThread->localRows; i++) {
        xNew[dataThread->startRow + i] = x[dataThread->startRow + i] - tay * residualVector[i];
    }

    int recvcounts[dataThread->blockSize];
    int displs[dataThread->blockSize];

    for (int i = 0; i < dataThread->blockSize; i++) {
        recvcounts[i] = dataThread->blockSize + (i < dataThread->extra ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, xNew, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    memcpy(x, xNew, N * sizeof(double));
}

void checkResult(double* x, double* u, int N) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(x[i] - u[i]) < EPSILON) {
            count++;
        }
    }
    if (count == N) {
        printf("all good\n");
    } else {
        printf("all bad\n");
    }

}

void solveBySimpleIteration(char** argv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int N = atoi(argv[1]);

    MatrixSet matrices;
    DataThreads dataThread;

    initializeDataThread(&dataThread, N, size, rank);
    memoryAllocation(&matrices, N, &dataThread);
    initializeMatrix(matrices.A, N, &dataThread); 
    initializeSolutionVector(matrices.u, N); 
    initializeBRandomVector(matrices.A, matrices.b, matrices.bRoot, matrices.u, N, &dataThread);
    initializeZeroVector(matrices.x, N); 
    initializeZeroVector(matrices.residualVector, N / size + (rank < N % size)); 
    
    double tay = calculateTAY(matrices.A, N, &dataThread);
    double residual = computeRelativeResidualNorm(matrices.residualVector, matrices.A, matrices.x, matrices.b, N, &dataThread);

    int iteration = 0;
    while (residual > EPSILON && iteration < 20000) {
        calculateNewIteration(matrices.residualVector, matrices.A, matrices.x, matrices.xNew, matrices.b, tay, N, &dataThread);
        residual = computeRelativeResidualNorm(matrices.residualVector, matrices.A, matrices.x, matrices.b, N, &dataThread);
        iteration++;
    } 

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Total execution time: %f seconds\n", elapsed_time);
        printf("Iteration %d, residual - %f\n", iteration, residual);
        checkResult(matrices.x, matrices.u, N);
    }

    matricesFree(&matrices);
}

int main(int argc, char** argv) {
    int mpi_status = MPI_Init(&argc, &argv);
    if (mpi_status != MPI_SUCCESS) {
        fprintf(stderr, "MPI Initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, mpi_status);
    } 

    solveBySimpleIteration(argv);

    MPI_Finalize();
    return 0;
}