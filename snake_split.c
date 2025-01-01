#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <string.h>

int readmatrix(int n, int** a, const char* filename)
{

    FILE *pf;
    pf = fopen (filename, "r");
    if (pf == NULL)
        return 0;

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            if (fscanf(pf, "%d", &a[i][j]) != 1) {
                fclose(pf);
                return 0;
            }

    fclose (pf); 
    return 1; 
}

int cmp_asc(const void *va, const void *vb)
{
  int a = *(int *)va, b = *(int *) vb;
  return a < b ? -1 : a > b ? +1 : 0;
}

int cmp_desc(const void *va, const void *vb)
{
  int a = *(int *)va, b = *(int *) vb;
  return a < b ? +1 : a > b ? -1 : 0;
}

void transpose(int **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            int temp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = temp;
        }
    }
}


void shearSort(int **A, int n) {
    int d = (int)ceil(log2(n)); // Calculate the number of steps
    for (int l = 1; l <= d + 1; l++) {
        for (int k = 0; k < n; k += 2) {
            // Sort odd rows in ascending order
            qsort(A[k], n, sizeof(int), cmp_asc); 
        }
        for (int k = 1; k < n; k += 2) {
            // Sort even rows in descending order
            qsort(A[k], n, sizeof(int), cmp_desc); 
        }
        if (l <= d) {
            for (int k = 0; k < n; k++) {
                // Sort all columns in ascending order
                transpose(A, n);
                qsort(A[k], n, sizeof(int), cmp_asc);
                transpose(A, n);
            }
        }
    }
}

void printMatrix(int **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }
}

// Function to print a flattened matrix
void printMatrixFlat(int *A, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d\t", A[i * m + j]);
        }
        printf("\n");
    }
}

void printArray(int *A, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d\t", A[i]);
    }
    printf("\n");
}

bool isMatrixSorted(int *A, int n) {
    // Check if rows are sorted in snake-like manner
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            int index = i * n + j;
            if (i % 2 == 0) {
                // Even-indexed rows should be sorted in ascending order
                if (A[index] > A[index + 1]) {
                    return false;
                }
            } else {
                // Odd-indexed rows should be sorted in descending order
                if (A[index] < A[index + 1]) {
                    return false;
                }
            }
        }
    }

    // Check if columns are sorted in ascending order
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n - 1; i++) {
            int index1 = i * n + j;
            int index2 = (i + 1) * n + j;
            if (A[index1] > A[index2]) {
                return false;
            }
        }
    }

    return true;
}

void Generate_random_matrix(int n, int** A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 100;
        }
    }
}

void transposeLocal(int *A, int n, int m, int *B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            B[j * n + i] = A[i * m + j];
        }
    }
}

void transposeMatrix(int* flatA, int n, int size, int rank, int *localA, int *localB) {
    // Scatter the flattened matrix to all processes
    int rows_per_proc = n > size ? n / size : 1;

    MPI_Scatter(flatA, n * rows_per_proc, MPI_INT, localA, n * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Transpose the local matrix
    transposeLocal(localA, rows_per_proc, n, localB);

    // Gather the transposed matrix to process 0
    for (int i = 0; i < n; i++) {
        MPI_Gather(localB + rows_per_proc*i, rows_per_proc, MPI_INT, flatA + n*i, rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void TransposedToNormal(int* localTranspose, int* localA, int rows_per_proc, int n, int size, int* counts, int* displs) {

    for (int p = 0; p < size; p++) {
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < size; j++) {
                counts[j] = rows_per_proc;
                displs[j] = rows_per_proc*j * rows_per_proc + i*rows_per_proc;
            }

            MPI_Scatterv(localTranspose, counts, displs, MPI_INT, localA + i*n + p*rows_per_proc, n * rows_per_proc, MPI_INT, p, MPI_COMM_WORLD);
        }
    }
}

void TransposedToOddEven(int* localTranspose, int* localA, int rows_per_proc, int n, int size, int* counts, int* displs) {

    for (int p = 0; p < size; p++) {
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < size; j++) {
                counts[j] = rows_per_proc;
                if (j % 2 == 0)
                    displs[j] = (j * rows_per_proc + i*2)*rows_per_proc;
                else
                    displs[j] = displs[j - 1] + rows_per_proc;
            }

            MPI_Scatterv(localTranspose, counts, displs, MPI_INT, localA + i*n + p*rows_per_proc, n * rows_per_proc, MPI_INT, p, MPI_COMM_WORLD);
        }
    }
}

void SortRows(int* localA, int n, int rows_per_proc, int (*cmp1)(const void*, const void*), int (*cmp2)(const void*, const void*)) {
    
    // Sort odd rows
    for (int k = 0; k < rows_per_proc; k += 2) {
        qsort(localA + k * n, n, sizeof(int), cmp1); 
    }

    // Sort even rows
    for (int k = 1; k < rows_per_proc; k += 2) {
        qsort(localA + k * n, n, sizeof(int), cmp2); 
    }

}

void OddEvenToNormal(int* localA, int* localB, int rows_per_proc, int n, int size,int pair, int rank) {
    for (int i = 0; i < rows_per_proc / 2; i++) {
        if (rank % 2 == 0) {
            // Copy to even rows of loalB
            memcpy(localB + 2 * i * n, localA + i * n, n * sizeof(int));
        } else {
            // Copy to odd rows of localB
            memcpy(localB + (2 * i + 1) * n, localA + (i + rows_per_proc/2) * n, n * sizeof(int));
        }
    }

    for (int i = 0; i < rows_per_proc / 2; i++) {
        if (rank % 2 == 0) {
            // Send to pair
            MPI_Send(localA + (i + rows_per_proc/2) * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD);
            // Receive from pair
            MPI_Recv(localB + (2 * i + 1) * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // Receive from pair
            MPI_Recv(localB + 2 * i * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Send to pair
            MPI_Send(localA + i * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD);
        }
    }
}

void NormalToOddEven(int* localA, int* localB, int rows_per_proc, int n, int size, int pair, int rank) {
    for (int i = 0; i < rows_per_proc / 2; i++) {
        if (rank % 2 == 0) {
            // Copy to even rows of localB
            memcpy(localB + i * n, localA + 2* i * n, n * sizeof(int));
        } else {
            // Copy to odd rows of localB
            memcpy(localB + (i + rows_per_proc/2) * n, localA + (2 * i + 1) * n, n * sizeof(int));
        }
    }

    for (int i = 0; i < rows_per_proc / 2; i++) {
        if (rank % 2 == 0) {
            // Send to pair
            MPI_Send(localA + (2 * i + 1) * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD);
            // Receive from pair
            MPI_Recv(localB + (i + rows_per_proc/2) * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // Receive from pair
            MPI_Recv(localB + i * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Send to pair
            MPI_Send(localA + 2 * i * n, n, MPI_INT, pair, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char* input_name = argv[1];
    int n = atoi(argv[2]);
    int output = atoi(argv[3]);

    int **A = NULL;
    int *flatA = NULL;

    int rows_per_proc = n > size ? n / size : 1;

    if (rank == 0) {
        if (argc != 4) {
            printf("Usage: ./snakeserial input_file n output\n");
            return 1;
        }
        A = (int **)malloc(n * sizeof(int *));
        for (int i = 0; i < n; i++) {
            A[i] = (int *)malloc(n * sizeof(int));
        }
        // Generate random matrix if output is 2
        if (output == 2){ 
            Generate_random_matrix(n, A);
        } else {
            // Read matrix from file
            if (!readmatrix(n, A, input_name)) {
                printf("Error reading matrix from file\n");
                return 1;
            }
        }

        // Print original matrix if output is 1 or 2
        if (output > 0){
            printf("Original Matrix:\n");
            printMatrix(A, n);
            printf("\n");
        }

        // Flatten the matrix A for MPI communication
        flatA = (int *)malloc(n * n * sizeof(int));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flatA[i * n + j] = A[i][j];
            }
        }
        
    }

    // Start timer
	double start_time = MPI_Wtime();


    /* _______________________________ 
    |                                |
    |  Shear sort                    |
    | ______________________________| */

    int *localA = (int *)malloc(n * rows_per_proc * sizeof(int));
    int *localB = (int *)malloc(n * rows_per_proc * sizeof(int));
    int* localTranspose = (int *)malloc(n * rows_per_proc * sizeof(int));

    int counts[size], displs[size];

    // Calculate the color for odd and even processes
    int color = (rank % 2); 
    int pair = !color ? rank + 1 : rank - 1;

    // Scatter the matrix by odd and even rows
    MPI_Scatter(flatA, n * rows_per_proc, MPI_INT, localA, n * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    NormalToOddEven(localA, localB, rows_per_proc, n, size, pair, rank);    

    int d = (int)ceil(log2(n)); // Calculate the number of steps
    for (int l = 1; l <= d + 1; l++) {

        if (color){ // Sort in descending order
            for (int i = 0; i < rows_per_proc; i++) {
                qsort(localB + i * n, n, sizeof(int), cmp_desc);
            }
        } else { // Sort in ascending order
            for (int i = 0; i < rows_per_proc; i++) {
                qsort(localB + i * n, n, sizeof(int), cmp_asc);
            }
        }

        OddEvenToNormal(localB, localA, rows_per_proc, n, size, pair, rank);

        if (l <= d) {

            // Transpose the matrix locally 
            transposeLocal(localA, rows_per_proc, n, localTranspose);

            // Gather the transposed submatrix to localA
            TransposedToNormal(localTranspose, localA, rows_per_proc, n, size, counts, displs);

            for (int k = 0; k < rows_per_proc; k++) {
                qsort(localA + k * n, n, sizeof(int), cmp_asc);
            }
            
            // Transpose back
            transposeLocal(localA, rows_per_proc, n, localTranspose);

            TransposedToOddEven(localTranspose, localB, rows_per_proc, n, size, counts, displs);

        }

    }

    // Gather the matrix
    MPI_Gather(localA, n * rows_per_proc, MPI_INT, flatA, n * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    free(localA);
    free(localB);
    free(localTranspose);


    // Get maximum time
    double max_time;
    double execution_time = MPI_Wtime() - start_time;
    MPI_Reduce(&execution_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    // Printing sorted matrix if output is 1 or 2
    if (rank == 0 && output > 0){
        printf("Sorted Matrix:\n");
        printMatrixFlat(flatA, n, n);
        printf("\n");

        // Check if the matrix is sorted correctly
        bool isSorted = isMatrixSorted(flatA, n);

        if (isSorted) {
            printf("Matrix is sorted correctly\n");
        } else {
            printf("Matrix is not sorted correctly\n");
        }
    }


    if (rank == 0) {
        printf("%f\n", max_time);
        free(flatA);
    }

    MPI_Finalize();

    return 0;
}


