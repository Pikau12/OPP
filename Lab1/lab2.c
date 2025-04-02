#include "lab2.h"

void printMatrix(double* A, int SS, int N, int rank) {
    printf("LLLOL - %d\n" , rank);
    for (int i = 0; i < SS; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", A[i*N + j]);
        }
        printf("  - %d \n", rank);
    }
}

void printVector(double* vect, int N, int rank) {
    for (int i = 0; i < N; i++) {
        printf("%lf - %d, %d\n", vect[i], rank, rank + 2*i);
    }
    printf("\n");
    printf("\n");
}

void checkResult(double* A, double* B, int N) {
    int counter = 0;
    for(int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) <= 0.01) {
            counter += 1;
        }
    }

    if(counter == N) { printf("All good\n"); }
    else { printf("All bad\n"); }
}