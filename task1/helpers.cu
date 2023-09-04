#include <stdio.h>
#include <math.h>

// return a random float number between a and b
float random_float(float a, float b) {
    return a + (b - a)*((float)rand()/(float)(RAND_MAX));
}

// fill matrices A and B with random values and fill C zeros
void fill_matrices(float *A, float *B, float *C, int M, int N, int P) {
    srand(time(0));
	for (int j=0; j<N; j++) {
		for (int i=0; i<M; i++)
			A[i*N + j] = random_float(3, 5);
		for (int i=0; i<P; i++)
			B[i*N + j] = random_float(2, 3);
	}
	for (int i=0; i<M; i++)
		for (int j=0; j<P; j++)
			C[i*P + j] = 0.0f;
}

// print MxN matrix
void print_matrix(float *matrix, int M, int N) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++)
			printf("%.2f ", matrix[i*N + j]);
		printf("\n");
	}
}

// check if C = AB has been done correctly
bool check_matmul(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float dot_prod = 0.0f;
            for (int k = 0; k < N; k++)
                dot_prod += A[i*N + k] * B[k*P + j];
            if (abs(C[i*P + j] - dot_prod) > 0.1)
                return false;
        }
    }
    return true;
}