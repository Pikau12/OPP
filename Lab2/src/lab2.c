#include "lab2.h"

void matricesFree(MatrixSet matrices, int N) {
    deallocateMatrix(matrices.A, N);
    free(matrices.b);
    free(matrices.x);
    free(matrices.u);
    free(matrices.residualVector);

    matrices.A = NULL;
    matrices.u = NULL;
    matrices.b = NULL;
    matrices.x = NULL;
    matrices.residualVector = NULL;
}

void deallocateMatrix(double** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix(double** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
}

void printVector(double* vect, int N) {
    for (int i = 0; i < N; i++) {
        printf("%lf \n", vect[i]);
    }
    printf("\n");
    printf("\n");
}

void checkResult(double* A, double* B, int N) {
    int counter = 0;
    for(int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) <= 0.001) {
            counter += 1;
        }
    }

    if(counter == N) { printf("All good\n"); }
    else { printf("All bad\n"); }
}