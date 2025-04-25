#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>

#define _USE_MATH_DEFINES
#define MIN_COUNT_ARGS 2
#define EPSILON 0.00000001
#define PARAMETR_A 100000
#define DX 2
#define DY 2
#define DZ 2
#define X0 -1
#define Y0 -1
#define Z0 -1
#define Nx 320
#define Ny 320
#define Nz 320

const double hx = (double)DX / (Nx - 1.0);
const double hy = (double)DY / (Ny - 1.0);
const double hz = (double)DZ / (Nz - 1.0);
const double invA = 1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + PARAMETR_A);

typedef struct {
    double* phi;
    double* previousPhi;
    double* downLayer;
    double* upLayer;
    int layerHeight;
} Matrix;

void memoryAllocation(Matrix* matrices, int size) {
    matrices->phi = (double*)malloc(sizeof(double) * Nx * Ny * Nz / size);
    matrices->previousPhi = (double*)malloc(sizeof(double) * Nx * Ny * Nz / size);
    matrices->downLayer = (double*)malloc(sizeof(double) * Nx * Ny);
    matrices->upLayer = (double*)malloc(sizeof(double) * Nx * Ny);

    if (!matrices->phi || !matrices->previousPhi || !matrices->downLayer || !matrices->upLayer) {
        fprintf(stderr, "Memory allocation failed in process\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void matricesFree(Matrix* matrices) {
    free(matrices->phi);
    free(matrices->previousPhi);
    free(matrices->downLayer);
    free(matrices->upLayer);

    *matrices = (Matrix) {0};
}

double phiFunction(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z) {
    return 6 - PARAMETR_A * phiFunction(x, y, z);
}

static inline int index3D(int x, int y, int z) {
    return z * (Nx * Ny) + y * Nx + x;
}

void initializeData(Matrix* matrices, int rank, int size) {
    matrices->layerHeight = Nz / size;
    for (int z = 0; z < matrices->layerHeight; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, z);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + (z + matrices->layerHeight * rank) * hz;
                if (y == 0 || x == 0 || y == Ny - 1 || x == Nx - 1) {
                    matrices->phi[idx] = phiFunction(x_coord, y_coord, z_coord);
                    matrices->previousPhi[idx] = phiFunction(x_coord, y_coord, z_coord);
                } else {
                    matrices->phi[idx] = 0.0;
                    matrices->previousPhi[idx] = 0.0;
                }
            }
        }
    }
    
    if (rank == 0) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, 0);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0;
                matrices->phi[idx] = phiFunction(x_coord, y_coord, z_coord);
                matrices->previousPhi[idx] = phiFunction(x_coord, y_coord, z_coord);
            }
        }
    }

    if (rank == size - 1) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, matrices->layerHeight - 1);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + DZ;
                matrices->phi[idx] = phiFunction(x_coord, y_coord, z_coord);
                matrices->previousPhi[idx] = phiFunction(x_coord, y_coord, z_coord);
            }
        }
    }
}

double JacobiMethod(Matrix* matrices, int rank, int x, int y, int z) {
    double xComp = (matrices->previousPhi[index3D(x - 1, y, z)] + matrices->previousPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (matrices->previousPhi[index3D(x, y - 1, z)] + matrices->previousPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (matrices->previousPhi[index3D(x, y, z - 1)] + matrices->previousPhi[index3D(x, y, z + 1)]) / (hz * hz);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + (z + matrices->layerHeight * rank) * hz;
    return invA * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

double JacobiMethodTopEdge(Matrix* matrices, int rank, int x, int y) {
    int z = matrices->layerHeight - 1;
    double xComp = (matrices->previousPhi[index3D(x - 1, y, z)] + matrices->previousPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (matrices->previousPhi[index3D(x, y - 1, z)] + matrices->previousPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (matrices->previousPhi[index3D(x, y, z - 1)] + matrices->upLayer[y * Nx + x]) / (hz * hz);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + ((matrices->layerHeight - 1) + matrices->layerHeight * rank) * hz;
    return invA * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

double JacobiMethodBottomEdge(Matrix* matrices, int rank, int x, int y) {
    int z = 0;
    double xComp = (matrices->previousPhi[index3D(x - 1, y, z)] + matrices->previousPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (matrices->previousPhi[index3D(x, y - 1, z)] + matrices->previousPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (matrices->downLayer[y * Nx + x] + matrices->previousPhi[index3D(x, y, z + 1)]) / (hz * hz);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + (matrices->layerHeight * rank) * hz;
    return invA * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

void CalculateCenter(Matrix* matrices, int rank, bool* flag) {
    for (int z = 1; z < matrices->layerHeight - 1; ++z) {
        for (int y = 1; y < Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                int idx = index3D(x, y, z);
                matrices->phi[idx] = JacobiMethod(matrices, rank, x, y, z);
                if (fabs(matrices->phi[idx] - matrices->previousPhi[idx]) > EPSILON) {
                    *flag = false;
                }
            }
        }
    }
}

void CalculateEdges(Matrix* matrices, int rank, bool* flag, int size) {
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            if (rank != 0) {
                int idx = index3D(x, y, 0);
                matrices->phi[idx] = JacobiMethodBottomEdge(matrices, rank, x, y);
                if (fabs(matrices->phi[idx] - matrices->previousPhi[idx]) > EPSILON) {
                    *flag = false;
                }
            }

            if (rank != size - 1) {
                int idx = index3D(x, y, matrices->layerHeight - 1);
                matrices->phi[idx] = JacobiMethodTopEdge(matrices, rank, x, y);
                if (fabs(matrices->phi[idx] - matrices->previousPhi[idx]) > EPSILON) {
                    *flag = false;
                }
            }
        }
    }
}

void CalculateMaxDifference(Matrix* matrices, int rank) {
    double maxDiff = 0.0;
    for (int z = 0; z < matrices->layerHeight; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + (z + matrices->layerHeight * rank) * hz;
                double diff = fabs(matrices->previousPhi[index3D(x, y, z)] - phiFunction(x_coord, y_coord, z_coord));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
    }
    double globalMax = 0.0;
    MPI_Allreduce(&maxDiff, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max difference: %f\n", globalMax);
    }
}

int main(int argc, char** argv) {
    int mpi_status = MPI_Init(&argc, &argv);
    if (mpi_status != MPI_SUCCESS) {
        fprintf(stderr, "MPI Initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, mpi_status);
    } 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix matrices;
    double startTime;
    if (rank == 0) {
        startTime = MPI_Wtime();
    }
    memoryAllocation(&matrices, size);
    initializeData(&matrices, rank, size);
    

    bool flag = false;

    int counter = 0;
    MPI_Request requests[4];

    while (!flag) {
        flag = true;    
        double* temp = matrices.phi;
        matrices.phi = matrices.previousPhi;
        matrices.previousPhi = temp;

        int reqCount = 0;

        if (rank != 0) {
            MPI_Isend(matrices.previousPhi, Nx * Ny, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &requests[reqCount++]);
            MPI_Irecv(matrices.downLayer, Nx * Ny, MPI_DOUBLE, rank - 1, 20, MPI_COMM_WORLD, &requests[1]);
        }
        if (rank != size - 1) {
            MPI_Isend(matrices.previousPhi + (matrices.layerHeight - 1) * Nx * Ny, Nx * Ny, MPI_DOUBLE, rank + 1, 20, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(matrices.upLayer, Nx * Ny, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &requests[3]);
        }

        CalculateCenter(&matrices, rank, &flag);

        // MPI_Waitall(reqCount, requests, MPI_STATUSES_IGNORE);

        if (rank != 0) {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        }
        if (rank != size - 1) {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

        CalculateEdges(&matrices, rank, &flag, size);

        bool globalFlag;
        MPI_Iallreduce(&flag, &globalFlag, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        flag = globalFlag;
        counter++;
    }
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;
    CalculateMaxDifference(&matrices, rank);

    if (rank == 0) {
        printf("Total execution time: %f seconds\n", elapsedTime);
        printf("Iterations: %d\n", counter);
    }
    matricesFree(&matrices);
    MPI_Finalize();
    return 0;
}