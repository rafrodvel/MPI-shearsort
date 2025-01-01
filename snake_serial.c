#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

static double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
    return seconds;
}

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

void sortColumn(int **A, int n) {
    for (int j = 0; j < n; j++) {
        int* temp = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            temp[i] = A[i][j];
        }
        qsort(temp, n, sizeof(int), cmp_asc);
        for (int i = 0; i < n; i++) {
            A[i][j] = temp[i];
        }
        free(temp);
    }
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
            transpose(A, n);
            for (int k = 0; k < n; k++) {
                // Sort all columns in ascending order
                qsort(A[k], n, sizeof(int), cmp_asc);
            }
            transpose(A, n);
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

bool isMatrixSorted(int **A, int n) {
    bool isSorted = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++){ 
            if (i % 2 == 0) {
                if (A[i][j] > A[i][j + 1]) isSorted = false;
            }
            else {
                if (A[i][j] < A[i][j + 1]) isSorted = false;
            }
        }
    }
    return isSorted;
}

void Generate_random_matrix(int n, int** A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 100;
        }
    }
}

int main(int argc, char *argv[]) {

    char* input_name = argv[1];
    int n = atoi(argv[2]);
    int print = atoi(argv[3]);
    int **A = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        A[i] = (int *)malloc(n * sizeof(int));
    }

    if (argc != 4) {
        printf("Usage: ./serial input_file n print\n");
        return 1;
    }

    // Generate random matrix if output is 2
    if (print == 2){ 
        Generate_random_matrix(n, A);
    } else {
        // Read matrix from file
        if (!readmatrix(n, A, input_name)) {
            printf("Error reading matrix from file\n");
            return 1;
        }
    }

    // Example: Fill the matrix with some values
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 50;
        } 

    if (print) {
        printf("Original Matrix:\n");
        printMatrix(A, n);
        printf("\n");
    }

    // Start timer
    double start_time = get_wall_seconds();

    shearSort(A, n);

    // Stop timer
    double end_time = get_wall_seconds();
    double elapsed_time = end_time - start_time;
    printf("%f\n", elapsed_time);

    if (print) {
        printf("Sorted Matrix:\n");
        printMatrix(A, n);
        printf("\n");
    }

    if (print){         
        // Check if the matrix is sorted correctly
        bool isSorted = isMatrixSorted(A, n);

        if (isSorted) {
            printf("\nMatrix is sorted correctly\n");
        } else {
            printf("\nMatrix is not sorted correctly\n");
        }
    }


    // Freeing dynamically allocated memory
    for (int i = 0; i < n; i++)
        free(A[i]);

    free(A);

    return 0;
}



