#include "helpers.cu"
#define M 3
#define N 1000
#define P 4

__global__ void matmul_outerprod(float* A, float* B, float* C) {
    __shared__ float A_col[N];
    __shared__ float B_row[N];
    __shared__ float outer_prod[M][P];

    // load to shared memroy
    A_col[threadIdx.x] = A[threadIdx.x*N + blockIdx.x]; // A[threadIdx.x][blockIdx.x]
    B_row[threadIdx.y] = B[blockIdx.x*P + threadIdx.y]; // B[blockIdx.x][threadIdx.y]
    __syncthreads();

    // calculate outer products
    outer_prod[threadIdx.x][threadIdx.y] = A_col[threadIdx.x] * B_row[threadIdx.y];
    __syncthreads();

    // sum the outer products
    atomicAdd(&C[threadIdx.x * P + threadIdx.y], outer_prod[threadIdx.x][threadIdx.y]);
}

int main() {
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

    dim3 threadsPerBlock(M, P);
    dim3 blocksPerGrid(N);
    matmul_outerprod<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev);

    cudaMemcpy(C, C_dev, M*P*sizeof(float), cudaMemcpyDeviceToHost);
    //! for debugging [begin]
    // printf("C\n");
    // print_matrix(C, M, P);
    //! for debugging [end]
    printf(check_matmul(A, B, C, M, N, P) ? "Test PASSED\n" : "Test FAILED\n");
    return 0;
}