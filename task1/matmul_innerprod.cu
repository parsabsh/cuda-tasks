#include "helpers.cu"

__global__ void matmul_innerprod(float *A, float *B, float *C, int M, int N, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float dot_product = 0.0f;  // use of local memory to reduce access to global memory
    for (int k = 0; k < N; k++)
        dot_product += A[i * N + k] * B[k * P + j];
    C[i * P + j] = dot_product;
}

int main() {
    int M = 3;
    int N = 1000;
    int P = 4;
    float *A = (float*) malloc(M*N*sizeof(float));
    float *B = (float*) malloc(N*P*sizeof(float));
    float *C = (float*) malloc(M*P*sizeof(float));
    fill_matrices(A, B, C, M, N, P);
    //! for debugging [begin]
    // printf("A\n");
    // print_matrix(A, M, N);
    // printf("B\n");
    // print_matrix(B, N, P);
    //! for debugging [end]
    float *A_dev, *B_dev, *C_dev;
    cudaMalloc((void**) &A_dev, M*N*sizeof(float));
    cudaMalloc((void**) &B_dev, N*P*sizeof(float));
    cudaMalloc((void**) &C_dev, M*P*sizeof(float));

    cudaMemcpy(A_dev, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, N*P*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev, C, M*P*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (P + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_innerprod<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev, M, N, P);

    cudaMemcpy(C, C_dev, M*P*sizeof(float), cudaMemcpyDeviceToHost);
    //! for debugging [begin]
    // printf("C\n");
    // print_matrix(C, M, P);
    //! end for debugging [end]
    printf(check_matmul(A, B, C, M, N, P) ? "Test PASSED\n" : "Test FAILED\n");
    return 0;
}
