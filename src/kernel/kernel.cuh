#pragma once
#include <math_constants.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
template <typename T>
__global__ void addKernel(T *data, T *grad, float lr, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        T res = data[i * N + j];
        T grad_val = lr*grad[i * N + j];
        data[i * N + j] = res-grad_val;
    }
}

template<typename T>
__global__ void copyKernel(T* src, T* dst, int nums)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<nums) {
        dst[idx] =src[idx];
    }
}

template<typename T, int block_size=32, int num_per_thread=4>
__global__ void GEMM(const T* W, const T* A, const T* b, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SW_x = idx%TILE;
    int SW_y = idx/TILE;
    int SA_x = idx%(block_size*num_per_thread);
    int SA_y = idx/(block_size*num_per_thread);
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]: 0;
        SA[idx] = i*TILE+SA_y<K&&SA_x+indB<N? A[(i*TILE+SA_y)*N+SA_x+indB]: 0;
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t+(j/num_per_thread+threadIdx.y*num_per_thread)*TILE]*SA[t*block_size*num_per_thread+threadIdx.x*num_per_thread+j%num_per_thread];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j]+b[row*N+col];
        }
    }
};

template<typename T, int block_size=32, int num_per_thread=4>
__global__ void mulKernel(const T* W, const T* A, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SW_x = idx%TILE;
    int SW_y = idx/TILE;
    int SA_x = idx%(block_size*num_per_thread);
    int SA_y = idx/(block_size*num_per_thread);
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]: 0;
        SA[idx] = i*TILE+SA_y<K&&SA_x+indB<N? A[(i*TILE+SA_y)*N+SA_x+indB]: 0;
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t+(j/num_per_thread+threadIdx.y*num_per_thread)*TILE]*SA[t*block_size*num_per_thread+threadIdx.x*num_per_thread+j%num_per_thread];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j];
        }
    }
};

template<typename T>
__global__ void mulKernel_native(const T* W, const T* A, const T* b, T* Y, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M&&col<N)
    {
        T sum = 0;
        for(int i=0;i<K;i++)
        {
            sum += W[row*K+i]*A[i*N+col];
        }
        Y[row*N+col] = sum+b[row*N+col];
    }
}

template<typename T, int block_size, int num_per_thread=4>
__global__ void mulTKernel(const T* W, const T* A, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SW_x = idx%TILE;
    int SW_y = idx/TILE;
    int SA_x = idx%TILE;
    int SA_y = idx/TILE;
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]:0.0;
        SA[idx] = SA_y+indB<N&&i*TILE+SA_x<K? A[i*TILE+SA_x+(SA_y+indB)*K]:0.0;
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t+(j/num_per_thread+threadIdx.y*num_per_thread)*TILE]*SA[t+(threadIdx.x*num_per_thread+j%num_per_thread)*TILE];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j];
        }
    }
};

template<typename T, int block_size, int num_per_thread=4>
__global__ void linearKernel(const T* W, const T* A, const T* b, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SW_x = idx%TILE;
    int SW_y = idx/TILE;
    int SA_x = idx%TILE;
    int SA_y = idx/TILE;
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]:0.0;
        SA[idx] = SA_y+indB<N&&i*TILE+SA_x<K? A[i*TILE+SA_x+(SA_y+indB)*K]:0.0;
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t+(j/num_per_thread+threadIdx.y*num_per_thread)*TILE]*SA[t+(threadIdx.x*num_per_thread+j%num_per_thread)*TILE];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j]+b[col];//因为b是一维向量，所以要进行广播
        }
    }
};

template<typename T, int block_size=32, int num_per_thread=4>
__global__ void TmulKernel(const T* W, const T* A, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SA_x = idx%(block_size*num_per_thread);
    int SA_y = idx/(block_size*num_per_thread);
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = TILE*i+SA_y<K&&indA+SA_x<M? W[(TILE*i+SA_y)*M+indA+SA_x]: 0.0;
        SA[idx] = i*TILE+SA_y<K&&SA_x+indB<N? A[(i*TILE+SA_y)*N+SA_x+indB]: 0.0;          
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t*block_size*num_per_thread+j/num_per_thread+threadIdx.y*num_per_thread]*SA[t*block_size*num_per_thread+threadIdx.x*num_per_thread+j%num_per_thread];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j];
        }
    }
};

// 参考自oneflow https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh
// 定义了模板函数 Inf()，并对不同数据类型进行特化
template <typename T>
__inline__ __device__ T Inf();
template <>
__inline__ __device__ float Inf<float>() {
return CUDART_INF_F;
}
template<>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a,b); }
};

//束内归约thread_group_width最大为32
//ReductionOp规定求和还是求最大值
template<template<typename> class ReductionOp, typename T, int thread_group_width>
__inline__ __device__ T warpReduce(T val){
    for(int offset = thread_group_width>>1; offset>0; offset>>=1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

//一维grid归约， blocksize:[32, 4], gridsize:[M/4]
template<typename T, int cols_per_thread>
__global__ void softmaxKernel(const T* src,  T* dst, int M, int N){
    const int thread_group_width = 32;
    int m_idx  = blockDim.y*blockIdx.x+threadIdx.y;
    T local_max = -Inf<T>() ;
    T local_sum = 0.0;
    T buf[cols_per_thread] = {0};
    //如果线程块没有铺满结果数组(M>gridDim.X*blockDim.y)，就需要循环计算 for(int idx = 0; idx<M; idx+=gridDim.X*blockDim.y)
    for(int row = m_idx; row<M; row+=gridDim.x*blockDim.y){
        for(int pack = 0; pack<cols_per_thread; pack++){
            int col = thread_group_width*pack+threadIdx.x;
            if(col<N){
                T tmp = src[row*N+col];
                local_max = max(local_max, tmp);
            }
        }
        local_max = warpReduce<MaxOp, T, thread_group_width>(local_max);
        for(int pack = 0; pack<cols_per_thread; pack++){
            int col = thread_group_width*pack+threadIdx.x;
            if(col<N){
                T tmp = src[row*N+col];
                buf[pack] = exp(tmp-local_max);
                local_sum += buf[pack];
            }
        }
        local_sum = warpReduce<SumOp, T, thread_group_width>(local_sum);
        for(int pack = 0; pack<cols_per_thread; pack++){
            int col = thread_group_width*pack+threadIdx.x;
            if(col<N){
                dst[row*N+col] = buf[pack]/local_sum;
            }
        }
    }
}

//sum 要求blocksize.x <=32
template<typename T>
__global__ void sumKernel(const T* src, T* dst, int M, int N, int dim){
    int num_target = dim==0? M : N;
    int target_featrue = dim==0? N : M;
    int num_per_thread = (target_featrue+31)/32;
    int row = blockIdx.x*blockDim.y+threadIdx.y;
    T local_sum = 0.0;
    if(row<num_target){
        for(int i = 0; i<num_per_thread; i++){
            int idx = i*32+threadIdx.x;
            if(idx<target_featrue){
                local_sum += dim==0? src[row*N+idx] : src[idx*N+row];
            }
        }
        local_sum = warpReduce<SumOp, T, 32>(local_sum);
        if(threadIdx.x==0){
            dst[row] = local_sum;
        }
    }
}