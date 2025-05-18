/*
This program is here to aid with optimizing matrix operations for ConvNet's since they require quite a lot operations during the Convolutional Layer.

I will be using a GEMM (General Matrix Multiplication) style approach to calculate matrix multiplication, which can be specified as: αAB + βC, where A and B are matrix inputs
α and β are scalar inputs and C is a pre-existing matrix that is being overwritten by the output matrix.

The following is a helpful explanation for how GEMM can be used in well known algorithms taken from the nvidia docs: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

A plain matrix product AB is a GEMM with α equal to one and β equal to zero. 
For example, in the forward pass of a fully-connected layer, the weight matrix would be argument A, incoming activations would be argument B, 
and α and β would typically be 1 and 0, respectively. 
*/


// To support parallelization I will be using AVX SIMD (Single Instruction Multiple Data) instructions for the CPU to be able to handle large matrix operations.
#include <immintrin.h>
#include <mm_malloc.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#define M 4
#define N 4
#define K 4

// Function to allocate aligned memory
float* allocate_aligned_matrix(size_t rows, size_t cols) {
    size_t alignment = 32; // 32-byte alignment for AVX
    size_t size = rows * cols * sizeof(float);
    return (float*)_mm_malloc(size, alignment);
}

// Function to free aligned memory
void free_aligned_matrix(float* matrix) {
    _mm_free(matrix);
}

// Function to initialize matrix with known values
void initialize_matrix(float* matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)(i + 1);
    }
}

// Function to print matrix
void print_matrix(const float* matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// GEMM implementation using AVX and FMA3
void gemm_avx_fma3(const float* A, const float* B, float* C, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 8) {
            __m256 c_vec = _mm256_load_ps(&C[i * n + j]);
            for (size_t p = 0; p < k; ++p) {
                __m256 a_elem = _mm256_set1_ps(A[i * k + p]);
                __m256 b_vec = _mm256_load_ps(&B[p * n + j]);
                c_vec = _mm256_fmadd_ps(a_elem, b_vec, c_vec);
            }
            _mm256_store_ps(&C[i * n + j], c_vec);
        }
    }
}

int main() {
    // Allocate aligned memory for matrices
    float* A = allocate_aligned_matrix(M, K);
    float* B = allocate_aligned_matrix(K, N);
    float* C = allocate_aligned_matrix(M, N);

    // Initialize matrices A and B
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    // Initialize matrix C to zero
    for (size_t i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    // Perform GEMM operation
    gemm_avx_fma3(A, B, C, M, N, K);

    // Output the result
    printf("Matrix A:\n");
    print_matrix(A, M, K);
    printf("\nMatrix B:\n");
    print_matrix(B, K, N);
    printf("\nMatrix C (Result of A * B):\n");
    print_matrix(C, M, N);

    // Free allocated memory
    free_aligned_matrix(A);
    free_aligned_matrix(B);
    free_aligned_matrix(C);

    return 0;
}
