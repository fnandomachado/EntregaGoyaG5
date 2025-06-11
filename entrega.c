#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/*
 * Matrix Multiplication Problem: Perform matrix multiplication using OpenMP.
 * Create two matrices A and B with dimensions NxM and MxP, respectively,
 * and initialize them with random values. Using OpenMP directives,
 * distribute the matrix multiplication among multiple threads.
 * Each thread is responsible for calculating a part of the resulting matrix.
 */

void initializeMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (double)rand() / RAND_MAX * 10.0; // Random value between 0 and 10
        }
    }
}

void multiplyMatrices(double *A, double *B, double *C, int N, int M, int P) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

void printMatrixPreview(double *matrix, int rows, int cols) {
    int preview_size = 3;
    printf("Matrix Preview (top-left %dx%d):\n", preview_size, preview_size);
    for (int i = 0; i < preview_size && i < rows; i++) {
        for (int j = 0; j < preview_size && j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Test cases: Small, Medium, Large
    int test_cases[3][3] = {
        {100, 100, 100},
        {500, 500, 500},
        {1000, 1000, 1000}
    };
    
    // Thread configurations to test
    int max_threads = omp_get_max_threads();
    int thread_counts[] = {1, 2, 4, max_threads};
    int num_thread_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    srand(time(NULL));
    
    printf("Maximum available threads: %d\n\n", max_threads);
    
    for (int test = 0; test < 3; test++) {
        int N = test_cases[test][0];
        int M = test_cases[test][1];
        int P = test_cases[test][2];
        
        printf("Test Case %d: Matrix A(%dx%d) * Matrix B(%dx%d)\n", test+1, N, M, M, P);
        
        // Allocate matrices
        double *A = (double*)malloc(N * M * sizeof(double));
        double *B = (double*)malloc(M * P * sizeof(double));
        double *C = (double*)malloc(N * P * sizeof(double));
        
        if (!A || !B || !C) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        
        // Initialize matrices with random values
        initializeMatrix(A, N, M);
        initializeMatrix(B, M, P);
        
        // Test with different thread counts
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            if (num_threads > max_threads) continue;
            
            omp_set_num_threads(num_threads);
            
            // Perform parallel matrix multiplication
            double start_time = omp_get_wtime();
            multiplyMatrices(A, B, C, N, M, P);
            double end_time = omp_get_wtime();
            
            printf("  Using %d threads: Completed in %.6f seconds\n", 
                   num_threads, end_time - start_time);
        }
        
        // Print preview of result matrix
        printMatrixPreview(C, N, P);
        printf("\n");
        
        // Free memory
        free(A);
        free(B);
        free(C);
    }
    
    return 0;
}