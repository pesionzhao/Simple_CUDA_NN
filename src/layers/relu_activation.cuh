#include "nn_layer.h"

template<typename T>
__global__ void reluForward(T* src, T* dst, int M, int N){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M&&col<N){
        dst[row*N+col] = max(0, src[row*N+col]);
    }   
}

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
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]:0;
        SA[idx] = i*TILE+SA_y<K&&SA_x+indB<N? A[(i*TILE+SA_y)*N+SA_x+indB]>0;
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
}
template<typename T>
class Relu: public NNLayer<T>{
    //y = max(0, x);
    std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> A){
        int M = A->rows;
        int N = A->cols;
        std::shared_ptr<Matrix<T>> dst = std::make_shared<Matrix<T>>(M, N)

    }
	std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dZ)
};