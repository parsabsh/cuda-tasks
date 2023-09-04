#include "helpers.cu"

__global__ void matmul_optimum(float *A, float *B, float *C, int M, int N, int P) {
    int c_index = blockIdx.x * P + blockIdx.y;  // C[blockIdx.x][blockIdx.y]
    int a_index = blockIdx.x * N + threadIdx.x;  // A[blockIdx.x][threadIdx.x]
    int b_index = threadIdx.x * P + blockIdx.y;  // B[threadIdx.x][blockIdx.y]
    atomicAdd(&C[c_index], A[a_index]*B[b_index]);
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

    dim3 blocksPerGrid(M, P, 1);
    dim3 threadsPerBlock(N, 1, 1);
    matmul_optimum<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev, M, N, P);

    cudaMemcpy(C, C_dev, M*P*sizeof(float), cudaMemcpyDeviceToHost);
    //! for debugging [begin]
    // printf("C\n");
    // print_matrix(C, M, P);
    //! for debugging [end]
    printf(check_matmul(A, B, C, M, N, P) ? "Test PASSED\n" : "Test FAILED\n");
    return 0;
}
