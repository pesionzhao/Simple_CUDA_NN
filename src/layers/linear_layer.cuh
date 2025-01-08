#pragma once
#include "nn_layer.h"
#include "../kernel/kernel.cuh"
//矩阵乘法 Y= W@A+b
//W: [M, K]
//A: [K, N]
//b: [M, N]
//Y: [M, N]
//索引重排，未使用float4
//TODO 更改共享内存排列方式防止bank冲突
template<typename T, int block_size=32, int num_per_thread=4>
__global__ void GEMMKernel(const T* W, const T* A, const T* b, T* Y, int M, int N, int K) {
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
        SW[idx] = W[(indA+SW_y)*K+TILE*i+SW_x];
        SA[idx] = A[(i*TILE+SA_y)*N+SA_x+indB];
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

//w@A
template<typename T, int block_size=32, int num_per_thread=4>
__global__ void backwardKernel(const T* W, const T* A, T* Y, int M, int N, int K) {
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
        SW[idx] = W[(TILE*i+SW_x)*M+indA+SW_y]; 
        SA[idx] = A[(i*TILE+SA_y)*N+SA_x+indB];//转置放入共享内存
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

//线性层: y = w@x+b
template<typename T>
class LinearLayer : public NNLayer<T> {
public:
    std::shared_ptr<Matrix<T>> W;
    std::shared_ptr<Matrix<T>> b;
    // Matrix<T> dW; //dL/dW
    // Matrix<T> db; //dL/db
    std::shared_ptr<Matrix<T>> Y;
    std::shared_ptr<Matrix<T>> A;
public:
    LinearLayer(int input_size, int output_size){
        W = std::make_shared<Matrix<T>>(output_size, input_size, this->train);
        b = std::make_shared<Matrix<T>>(output_size, 1, this->train);
        initWeights();
        initBias();
        this->params = {W, b};
    }
    std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> A){
        if(A->cols!=1)
            throw std::runtime_error("LinearLayer::forward: input matrix must be a column vector");
        this->A = A;
        Y = std::make_shared<Matrix<T>>(b->rows, 1);
        Y->allocate();
        const int num_per_thread = 1;
        const int block_size_x = 16;
        const int block_size_y = 16;
        int grid_size_x = (Y->cols + block_size_x*num_per_thread - 1) / (block_size_x*num_per_thread);
        int grid_size_y = (Y->rows + block_size_y*num_per_thread - 1) / (block_size_y*num_per_thread);
        int M = W->rows;
        int N = A->cols;
        int K = W->cols;
        if(W->cols!=A->rows){
            std::ostringstream oss;
            oss << "LinearLayer error: W->cols != A->rows which " 
                << W->cols << " != " << A->rows;
            throw std::runtime_error(oss.str()); // 抛出异常并传递格式化后的错误消息
        }
        mulKernel_native<T><<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(W->data_device.get(), A->data_device.get(), b->data_device.get(), Y->data_device.get(), M, N, K);
        // GEMMKernel<T, block_size_x><<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(W.data_device.get(), A.data_device.get(), b.data_device.get(), Y.data_device.get(), M, N, K);
        NNException::throwIfDeviceErrorsOccurred("GEMM error\n");
        return Y;

    }
    //链式法则求梯度，dz为后一层的梯度，逐步链到第一层, dY = dL / dY
    std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dY){
        if(!this->train)
            throw NNException("Linear layer is not in training mode");
        const int num_per_thread = 4;
        const int block_size_x = 32;
        const int block_size_y = 32;
        int grid_size_x = (A->cols + block_size_x*num_per_thread - 1) / (block_size_x*num_per_thread);
        int grid_size_y = (A->rows + block_size_y*num_per_thread - 1) / (block_size_y*num_per_thread);
        TmulKernel<T, block_size_x, num_per_thread><<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(W->data_device.get(), dY->data_device.get(), A->grad->data_device.get(), A->rows, A->cols, W->rows);
        NNException::throwIfDeviceErrorsOccurred("GEMM error occurs in linear layer backward for x\n");
        //计算其他梯度
        grid_size_x = (W->cols + block_size_x*num_per_thread - 1) / (block_size_x*num_per_thread);
        grid_size_y = (W->rows + block_size_y*num_per_thread - 1) / (block_size_y*num_per_thread);
        //dz为后一层传到上一层的梯度
        //对于y = w@a + b ==>> 储存梯度信息： dy/dw = a, dy/db = 1， 链式求导 dy/da = w.T * dz 
        mulTKernel<T, block_size_x, num_per_thread><<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(dY->data_device.get(), A->data_device.get(), W->grad->data_device.get(), W->rows, W->cols, dY->cols);
        NNException::throwIfDeviceErrorsOccurred("GEMM error occurs in linear layer backward for W\n");
        copyKernel<<<(int)(b->rows+63)/64, 64>>>(dY->data_device.get(), b->grad->data_device.get(), b->rows);
        NNException::throwIfDeviceErrorsOccurred("GEMM error occurs in linear layer backward for b\n");
        return A->grad;
    }

    void initWeights(unsigned long long seed=0){
        // W.randomInitDevice(seed);
        W->init_Xavier(seed);
        // W*=0.01;
    }
    void initBias(){
        b->zeroInitDevice();
    }
    void save_data() {
        W->save_to_file("W.bin");
    }

};
