#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define _USE_MATH_DEFINES
#define MIN_COUNT_ARGS 2
#define EPSILON 0.000001

typedef struct {
    double* A;
    double* u;
    double* b;
    double* x;
    double* residualVector;
    double* xBuffer;
} MatrixSet;

typedef struct {
    int blockSize;
    int extra;
    int startRow;
    int endRow;
    int localRows; 
} DataThreads;

void fillingLocalStrings(int* localRows, int N, int size) {
    int startRow = 0;
    int endRow = 0;
    for(int i = 0; i < size; i++) {
        startRow = i * N / size + (i < N % size ? i : N % size);
        endRow = startRow + N / size + (i < N % size ? 1 : 0);
        localRows[i] = endRow - startRow;
    }
}

void initializeDataThread(DataThreads* dataThread, int N, int size, int rank) {
    dataThread->blockSize = N / size;
    dataThread->extra = N % size;
    dataThread->startRow = rank * dataThread->blockSize + (rank < dataThread->extra ? rank : dataThread->extra);
    dataThread->endRow = dataThread->startRow + dataThread->blockSize + (rank < dataThread->extra ? 1 : 0);
    dataThread->localRows = dataThread->endRow - dataThread->startRow;

    
}

void matricesFree(MatrixSet* matrices) {
    free(matrices->A);
    free(matrices->b);
    free(matrices->x);
    free(matrices->u);
    free(matrices->residualVector);
    free(matrices->xBuffer);

    matrices->A = NULL;
    matrices->b = NULL;
    matrices->x = NULL;
    matrices->u = NULL;
    matrices->residualVector = NULL;
    matrices->xBuffer = NULL;
}

void memoryAllocation(MatrixSet* matrices, int N, DataThreads* dataThread) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    matrices->A = (double*)malloc(sizeof(double) * dataThread->localRows * N);
    matrices->u = (double*)malloc(sizeof(double) * N);
    matrices->b = (double*)calloc(dataThread->localRows, sizeof(double));
    matrices->x = (double*)calloc(dataThread->localRows, sizeof(double));
    matrices->xBuffer = (double*)malloc(sizeof(double) * ((N / size) + 1));
    matrices->residualVector = (double*)malloc(sizeof(double) * dataThread->localRows);

    if (!matrices->A || !matrices->u || !matrices->b || 
        !matrices->x || !matrices->xBuffer || !matrices->residualVector) {
        fprintf(stderr, "Memory allocation failed in process\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void initializeMatrix(double* matrix, int N, DataThreads* dataThread) {
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        int global_i = dataThread->startRow + local_i;
        for (int j = 0; j < N; j++) {
            matrix[local_i * N + j] = (j == global_i) ? 2.0 : 1.0;
        }
    }
}

void initializeBRandomVector(double* A, double* b, double* u, int N, DataThreads* dataThread) {
    for (int local_i = 0; local_i < dataThread->localRows; local_i++) {
        for (int j = 0; j < N; j++) {
            b[local_i] += A[local_i * N + j] * u[j];
        }
    }
}

void initializeZeroVector(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void initializeSolutionVector(double* u, int N) {
    for (int i = 0; i < N; i++) {
        u[i] = sin(2 * M_PI * i / N);        
    }
}

double calculateTAY(double* A, int N, DataThreads* dataThread) {
    double localMaxSum = 0.0;
    for (int i = 0; i < dataThread->localRows; i++) {
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

double computeRelativeResidualNorm(double* residualVector, double* A, double* x, double* xBuffer, double* b, int N, int size, int rank, int* allLocalRows, DataThreads* dataThread) {
    double local_r = 0.0, local_b = 0.0;
    memset(residualVector, 0, sizeof(double) * dataThread->localRows);
    memset(xBuffer, 0, sizeof(double) * dataThread->localRows);


    int rankOffset = 0;
    for (int r = 0; r < rank; r++) {
        rankOffset += allLocalRows[r];
    }
    for (int i = 0; i < dataThread->localRows; i++) {
        for (int j = 0; j < dataThread->localRows; j++) {
            residualVector[i] += A[i * N + (j + rankOffset)] * x[j];
        }
    }

    MPI_Request requests[2];

    for (int step = 0; step < size - 1; step++) {
        int sendTo = (rank + step + 1) % size;
        int recvFrom = (rank - step - 1 + size) % size;
        int sendCount = allLocalRows[rank];
        int recvCount = allLocalRows[recvFrom];
    
        MPI_Sendrecv(x, sendCount, MPI_DOUBLE, sendTo, 0,
                     xBuffer, recvCount, MPI_DOUBLE, recvFrom, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
        int recvOffset = 0;
        for (int r = 0; r < recvFrom; r++) {
            recvOffset += allLocalRows[r];
        }
    
        for (int i = 0; i < dataThread->localRows; i++) {
            for (int j = 0; j < recvCount; j++) {
                residualVector[i] += A[i * N + (j + recvOffset)] * xBuffer[j];
            }
        }
    }

    for (int i = 0; i < dataThread->localRows; i++) {
        local_r += (residualVector[i] - b[i]) * (residualVector[i] - b[i]);
        local_b += b[i] * b[i]; 
    }


    double global_r, global_b;
    MPI_Allreduce(&local_r, &global_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_b, &global_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_r/global_b);
}

void calculateNewIteration(double* residualVector, double* A, double* x, double* xBuffer, double* b, double tay, int N, int size, int rank, int* allLocalRows, DataThreads* dataThread) {
    memset(residualVector, 0, sizeof(double) * dataThread->localRows);
    memset(xBuffer, 0, sizeof(double) * dataThread->localRows);

    int rankOffset = 0;
    for (int r = 0; r < rank; r++) {
        rankOffset += allLocalRows[r];
    }
    for (int i = 0; i < dataThread->localRows; i++) {
        for (int j = 0; j < dataThread->localRows; j++) {
            residualVector[i] += A[i * N + (j + rankOffset)] * x[j];
        }
    }

    MPI_Request requests[2];

    // Обмен данными между процессами
    for (int step = 0; step < size - 1; step++) {
        int sendTo = (rank + step + 1) % size;
        int recvFrom = (rank - step - 1 + size) % size;
    
        int sendCount = dataThread->localRows;
        int recvCount = allLocalRows[recvFrom];
    
        MPI_Sendrecv(x, sendCount, MPI_DOUBLE, sendTo, 0,
                     xBuffer, recvCount, MPI_DOUBLE, recvFrom, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
        int recvOffset = 0;
        for (int r = 0; r < recvFrom; r++) {
            recvOffset += allLocalRows[r];
        }
    
        for (int i = 0; i < dataThread->localRows; i++) {
            for (int j = 0; j < recvCount; j++) {
                residualVector[i] += A[i * N + (j + recvOffset)] * xBuffer[j];
            }
        }
    }

    // Обновление x
    for (int i = 0; i < dataThread->localRows; i++) {
        double delta = tay * (residualVector[i] - b[i]);
        x[i] -= delta;
    }
}

void checkResult(double* x, double* u, int N, int rank, DataThreads* dataThread) {
    int count = 0;
    for (int i = 0; i < dataThread->localRows; i++) {
        int globalIndex = dataThread->startRow + i;

        if (fabs(x[i] - u[globalIndex]) < EPSILON + 1e-6) {
            count++;
        }
    }

    if (count == dataThread->localRows) {
        printf("rank %d - all good\n", rank);
    } else {
        printf("rank %d - all bad\n", rank);
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
    int allLocalRows[size];
    fillingLocalStrings(allLocalRows, N, size);

    initializeDataThread(&dataThread, N, size, rank);

    memoryAllocation(&matrices, N, &dataThread);
    initializeMatrix(matrices.A, N, &dataThread); 
    initializeSolutionVector(matrices.u, N); 
    initializeBRandomVector(matrices.A, matrices.b, matrices.u, N, &dataThread);
    initializeZeroVector(matrices.x, N / size + (rank < N % size)); 
    initializeZeroVector(matrices.residualVector, N / size + (rank < N % size)); 
    double tay = calculateTAY(matrices.A, N, &dataThread);


    double residual = 0.0;
    int iteration = 0;
    
    do {
        calculateNewIteration(matrices.residualVector, matrices.A, matrices.x, matrices.xBuffer, matrices.b, tay, N, size, rank, allLocalRows, &dataThread);
        residual = computeRelativeResidualNorm(matrices.residualVector, matrices.A, matrices.x, matrices.xBuffer, matrices.b, N, size, rank, allLocalRows, &dataThread);
        iteration++;
    }  while (residual > EPSILON && iteration < 20000);
 
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Total execution time: %f seconds\n", elapsed_time);
        printf("Iteration %d, residual - %f\n", iteration, residual);
    }

    checkResult(matrices.x, matrices.u, N, rank, &dataThread);

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