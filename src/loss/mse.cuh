#pragma once
#include "loss.cuh"
#include "../kernel/kernel.cuh"
//归约
//block = <32, 4>, grid = <M/4>, 一个线程要处理横向N/32个元素的归约，纵向4行的并行执行
template<typename T>
__global__ void mseKernel(T *predictions, T *target, T *per_block_res, int M, int N){
    int row = blockIdx.x*blockDim.y + threadIdx.y;//线程块是一维排布的，所以只有blockIdx.x和gridDim.x
    T local_sum = 0;
    //首先要将矩阵列归约为32列方便做束内规约
    int cols_per_thread = (N+31)/32;
    for(int i = 0; i<cols_per_thread; i++){
        int col = 32*i+threadIdx.x;
        if(col<N){
            T tmp = predictions[row*N+col] - target[row*N+col];
            local_sum += tmp*tmp;
        }
    }
    local_sum = warpReduce<SumOp, T, 32>(local_sum);
    if(threadIdx.x==0){
        per_block_res[row] = local_sum;
    }
}

//一维向量归约，固定blocksize = 32, grid = 1;
template<typename T>
__global__ void reduce(T* per_block_res, T* res, int N){
    T sum = 0;
    int cols_per_thread = (N+31)/32;
    for(int i = 0; i<cols_per_thread; i++){
        int col = 32*i+threadIdx.x;
        // printf("%d col element is %f\n" , col, per_block_res[col]);
        if(col<N)
            sum += per_block_res[col];
    }
    sum = warpReduce<SumOp, T, 32>(sum);
    if(threadIdx.x == 0) {
        res[0] = sum;
    }
}

template<typename T>
__global__ void mseBackward(T *predictions, T *target, T* grad, int M, int N){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    if(col<N && row<M){
        T p = predictions[row*N+col];
        T t = target[row*N+col];
        grad[row*N+col] = (p-t)/(M*N); //这里的梯度计算系数可以更改
    }
}

template <typename T>
class MSE : public Loss<T>{
private:
    float _cost;
public:
    float cost(std::shared_ptr<Matrix<T>> predictions, std::shared_ptr<Matrix<T>> target) override{
        this->predictions = predictions;
        this->target = target;
        int M = predictions->rows;
        int N = predictions->cols;
        T* row_res_device = nullptr;
        cudaMalloc((void**)&row_res_device, predictions->rows*sizeof(T));
        dim3 block_size(32, 32);
        int grid_size = (predictions->rows+block_size.y-1)/block_size.y;
        mseKernel<T><<<grid_size, block_size>>>(predictions->data_device.get(), target->data_device.get(), row_res_device, predictions->rows, target->cols);
        NNException::throwIfDeviceErrorsOccurred("mseKernel error in row reduce");
        T* res_device = nullptr;
        cudaMalloc((void**)&res_device, sizeof(T));
        reduce<T><<<1, 32>>>(row_res_device, res_device, predictions->rows);
        NNException::throwIfDeviceErrorsOccurred("mseKernel error in reduce");
        cudaFree(row_res_device);
        T* res_host = new T;
        cudaMemcpy(res_host, res_device, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(res_device);
        std::cout<<"mse: "<<*res_host/(M*N)<<"  "<<std::endl;
        return *res_host;
    };
    std::shared_ptr<Matrix<T>> dCost() override{
        int rows = this->predictions->rows;
        int cols = this->predictions->cols;
        std::shared_ptr<Matrix<T>> dY = std::make_shared<Matrix<T>>(rows, cols, false);
        dim3 block_size(32, 32);
        dim3 gird_size((cols + block_size.x - 1)/block_size.x, (rows + block_size.y - 1)/block_size.y);
        mseBackward<<<gird_size, block_size>>>(this->predictions->data_device.get(), this->target->data_device.get(), dY->data_device.get(), rows, cols);
        NNException::throwIfDeviceErrorsOccurred("mse backward error occurs");
        return dY;
    }
};