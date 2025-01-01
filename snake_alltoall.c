#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>

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

void transposeLocal(int *A, int total_col, int rows_per_proc, int rank, int size) {
    int temp;
    for (int k = 0; k < size; k++) {
        int col = k * rows_per_proc;
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = i + 1; j < rows_per_proc; j++) {
                temp = A[i * total_col + col + j];
                A[i * total_col + col + j] = A[j * total_col + col + i];
                A[j * total_col + col + i] = temp;
            }
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
        if (output > 1){ 
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

        // Freeing dynamically allocated memory
        for (int i = 0; i < n; i++)
            free(A[i]);

        free(A);
        
    }

    // Start timer
	double start_time = MPI_Wtime();


    /* _______________________________ 
    |                                |
    |  Shear sort                    |
    | ______________________________| */
    
    int rows_per_proc = n > size ? n / size : 1;

    int *localA = (int *)malloc(n * rows_per_proc * sizeof(int));
    int* localTranspose = (int *)malloc(n * rows_per_proc * sizeof(int));

    // Create a derived datatype to form chunks to distribute
    MPI_Datatype unaligned_chunk_type, chunk_type;
    MPI_Type_vector(rows_per_proc, rows_per_proc, n , MPI_INT, &unaligned_chunk_type);
    MPI_Type_create_resized(unaligned_chunk_type, 0, rows_per_proc * sizeof(int), &chunk_type);
    MPI_Type_free(&unaligned_chunk_type);
    MPI_Type_commit(&chunk_type);    

    // Scatter the matrix
    MPI_Scatter(flatA, n * rows_per_proc, MPI_INT, localA, n * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    int d = (int)ceil(log2(n)); // Calculate the number of steps
    for (int l = 1; l <= d + 1; l++) {

        if (rows_per_proc % 2 == 0){

            SortRows(localA, n, rows_per_proc, cmp_asc, cmp_desc);

        } else { // Odd numbers oscilate between ascending and descending
            if (rank % 2 == 0){
                
                SortRows(localA, n, rows_per_proc, cmp_asc, cmp_desc);

            }else{
                
                SortRows(localA, n, rows_per_proc, cmp_desc, cmp_asc);

                } 
            }

        if (l <= d) {
            // Transpose the local matrix
            transposeLocal(localA, n, rows_per_proc, rank, size);

            // Distribute to transpose globally
            MPI_Alltoall(localA, 1, chunk_type, localTranspose, 1, chunk_type, MPI_COMM_WORLD);

            // Sort the rows of transposed matrix
            for (int k = 0; k < rows_per_proc; k++) {
                qsort(localTranspose + k * n, n, sizeof(int), cmp_asc);
            }

            // Transpose back
            transposeLocal(localTranspose, n, rows_per_proc, rank, size);

            MPI_Alltoall(localTranspose, 1, chunk_type, localA, 1, chunk_type, MPI_COMM_WORLD);    

        }

    }

    // Gather the matrix
    MPI_Gather(localA, n * rows_per_proc, MPI_INT, flatA, n * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    free(localTranspose);
    free(localA);
    MPI_Type_free(&chunk_type);


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

