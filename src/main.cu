#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

constexpr int b_ = 1024;
constexpr int m_ = 1024;
constexpr int n_ = 1024;
constexpr int k_ = 1024;
constexpr int n_iter = 50;

int cublas_test(float* a, float* b, float* c) {
    float* d_a, *d_b, *d_c;
    size_t size_a = sizeof(float) * b_ * m_ * k_;
    size_t size_b = sizeof(float) * b_ * n_ * k_;
    size_t size_c = sizeof(float) * b_ * m_ * n_;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_c, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasStatus_t ret;
    ret = cublasCreate(&handle);

    float *a_array[b_], *b_array[b_];
    float *c_array[b_];
    for (int i = 0; i < b_; ++i) {
        a_array[i] = d_a + i * m_ * k_;
        b_array[i] = d_b + i * k_ * n_;
        c_array[i] = d_c + i * m_ * n_;
    }
    const float **d_Aarray, **d_Barray;
    float **d_Carray;
    cudaMalloc((void**)&d_Aarray, b_*sizeof(float *));
    cudaMalloc((void**)&d_Barray, b_*sizeof(float *));
    cudaMalloc((void**)&d_Carray, b_*sizeof(float *));
    cudaMemcpy(d_Aarray, a_array, b_*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, b_array, b_*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, c_array, b_*sizeof(float *), cudaMemcpyHostToDevice);

    const float alpha  =  1.0f;
    const float beta  =  0.0f;
    int m = m_;
    int n = n_;
    int k = k_;
    int lda = m_;
    int ldb = k_;
    int ldc = m_;
    int batch = b_;

    cout << "start cublasSgemmBatched benchmark" << endl;

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n_iter; i++) {
        ret = cublasSgemmBatched(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            m,n,k,
                            &alpha,
                            d_Aarray,  lda,
                            d_Barray,  ldb,
                            &beta,
                            d_Carray,  ldc,
                            batch);
    }
    gettimeofday(&end, NULL);

    // if (ret == CUBLAS_STATUS_SUCCESS) {
    //     printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
    // }
    printf("[INFO] batch_size %d m %d n %d k %d cublasSgemmBatched-time %.5f ms ( %d iterations) \n", b_, m_, n_, k_,
          ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / n_iter, n_iter);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    cudaFree(d_Carray);
    return 0;
};

int main() {

    float* a = new float[b_ * m_ * k_];
    for(int i = 0; i < b_ * m_ * k_; i++) a[i] = i;

    float* b = new float[b_ * k_ * n_];
    for(int i = 0; i < b_ * k_ * n_; i++) b[i] = i+1;

    float* c = new float[b_ * m_ * n_];
    for(int i = 0; i < b_ * m_ * n_; i++) c[i] = 0.0;

    cublas_test(a, b, c);

    return 0;
}