#include "lab2.h"
#include <time.h>


int main(int argc, char** argv) {

    if (argc < MIN_COUNT_ARGS) {
        fprintf(stderr, "Invalid argc: %d, must be more than 2\n", argc);
        return 1;
    }
    int countThreads = atoi(argv[2]);
    omp_set_num_threads(countThreads);

    solveBySimpleIteration(argv);

    return 0;
}

void solveBySimpleIteration(char** argv) {
    struct timespec start, end;

    // Засекаем время начала
    clock_gettime(CLOCK_REALTIME, &start);

    int N = atoi(argv[1]);
    if (N == 0) {
        fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
        return;
    }

    MatrixSet matrices;

    matrices.A = allocateMatrix(N);
    matrices.u = (double*)malloc(sizeof(double) * N);
    matrices.b = (double*)malloc(sizeof(double) * N);
    matrices.x = (double*)malloc(sizeof(double) * N);
    matrices.residualVector = (double*)malloc(sizeof(double) * N);

    initializeMatrix(matrices.A, N);
    initializeSolutionVector(matrices.u, N);
    initializeBRandomVector(matrices.A, matrices.b, matrices.u, N);
    initializeZeroVector(matrices.x, N);
    initializeZeroVector(matrices.residualVector, N);
    
    
    #pragma omp parallel
    {
        while (computeRelativeResidualNorm(matrices.residualVector, matrices.A, matrices.x, matrices.b, N) > EPSILON*EPSILON) {
            calculateNewIteration(matrices.residualVector, matrices.A, matrices.x, matrices.b, N);
        }
            
    }

    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Total time taken: %.9f seconds\n", elapsed);

    checkResult(matrices.u, matrices.x, N);

    matricesFree(matrices, N);

}

double** allocateMatrix(int N) {
    double** matrix = (double**)malloc(sizeof(double*) * N);
    for (int i = 0; i < N; i++) {
        matrix[i] = (double*)malloc(sizeof(double) * N);
    }
    return matrix;
}

void initializeMatrix(double** matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                matrix[i][j] = 2.0;
            } else {
                matrix[i][j] = 1.0;
            }
        }
    }
}

void initializeBRandomVector(double** A, double* b, double* u, int N) {
    matrixMult(b, A, u, N);
}

void initializeZeroVector(double* x, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 0;
    }
}

void initializeSolutionVector(double* u, int N) {
    for (int i = 0; i < N; i++) {
        u[i] = sin(2 * M_PI * i / N);
    }
}

double computeRelativeResidualNorm(double* residualVector, double** A, double* x, double* b, int N) {
    matrixMult(residualVector, A, x, N);
    vectorSub(residualVector, b, N);

    double upSum = 0, downSum = 0;
    #pragma omp for
    for (int i = 0; i < N; i++) {
        upSum += residualVector[i] * residualVector[i]; 
        downSum += b[i] * b[i];
    }

    return (upSum / downSum);
}

void matrixMult(double* residualVector, double** A, double* x, int N) {
    #pragma omp for schedule(dynamic, 10)
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * x[j];
        }
        residualVector[i] = sum;
    }
}

double calculateTAY(double** A, int N) {
    
    double maxSum = 0;
    
    #pragma omp for
    for (int i = 0; i < N; i++) {
        double rowSum = 0;
        for (int j = 0; j < N; j++) {
            rowSum += fabs(A[i][j]);
        }
        maxSum = fmax(maxSum, rowSum);
    }
    return 1.0 / maxSum; 
}

void vectorSub(double* residualVector, double* b, int N) {
    #pragma omp for
    for (int i = 0; i < N; i++) {
        residualVector[i] -= b[i];
    }
}

void scalarMult(double* residualVector, double scalar, int N) {
    #pragma omp for
    for (int i = 0; i < N; i++) {
        residualVector[i] *= scalar;
    }
}

void calculateNewIteration(double* residualVector, double** A, double* x, double* b, int N) {
    double tay = calculateTAY(A, N);
    matrixMult(residualVector, A, x, N);
    vectorSub(residualVector, b, N);
    scalarMult(residualVector, tay, N);
    vectorSub(x, residualVector, N);
}