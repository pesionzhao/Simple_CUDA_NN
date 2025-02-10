#pragma once

#include"nn_layer.h"

//假设src是一维向量, 公式推导见博客
template<typename T>
__global__ void softmaxJac(T* src, T* dst, int N){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    if(row<N&&col<N){
        T Si = src[row];
        T Sj = src[col];
        T tmp = Si == Sj ? Si: 0;
        dst[row*N+col] = tmp-Si*Sj;
    }
}

template<typename T>
std::shared_ptr<Matrix<T>> softmaxBackward(std::shared_ptr<Matrix<T>> softmax_output, std::shared_ptr<Matrix<T>> dy)
{
    const int block_x = 16;
    const int block_y = 16;
    int grid_x = (dy->cols+block_x-1)/block_x;
    int grid_y = (dy->rows+block_y-1)/block_y;
    std::shared_ptr<Matrix<T>> jac = std::make_shared<Matrix<T>>(dy->rows, dy->rows); // 输入输出大小一致，所以雅可比shape长宽一致
    std::shared_ptr<Matrix<T>> dx = std::make_shared<Matrix<T>>(dy->rows, dy->cols);
    softmaxJac<T><<<dim3(grid_y, grid_y), dim3(block_x, block_y)>>>(softmax_output->data_device.get(), jac->data_device.get(), dy->rows);
    TmulKernel<T, block_x, 1><<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(jac->data_device.get(), dy->data_device.get(), dx->data_device.get(), dy->rows, dy->cols, dy->rows);
    return dx;
}

//假设输入为一维列向量
template<typename T>
class Softmax: public NNLayer<T>{
public:
    std::shared_ptr<Matrix<T>> input; //shape(rows, 1)
    std::shared_ptr<Matrix<T>> output; //shape(rows, 1)
    Softmax(){}
    std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> input) override{
        if(input->cols!=1) //只能适用于一维列向量
            throw NNException("Currently Softmax layer is only suitable for one-dimensional vector");
        int N = input->rows;
        int M = input->cols;
        const int cols_per_thread = 1;
        if(cols_per_thread*32<N)
            throw NNException("Softmax cols_per_thread need be set larger");
        this->input = input;
        this->output = std::make_shared<Matrix<T>>(N, M);
        dim3 block_size(32, 4);
        int grid_size = (N+block_size.y-1)/block_size.y;
        softmaxKernel<T, cols_per_thread><<<grid_size, block_size>>>(input->data_device.get(), output->data_device.get(), M, N);
        if(this->train){
            output->op = "softmax";
            output->prev.insert(input);
            output->setBackward([this](std::shared_ptr<Matrix<T>> dy) {
                this->input->addGrad(softmaxBackward(this->output, dy));
            });
        }
        return this->output;
    }
    std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dy){
        const int block_x = 16;
        const int block_y = 16;
        int grid_x = (dy->cols+block_x-1)/block_x;
        int grid_y = (dy->rows+block_y-1)/block_y;
        std::shared_ptr<Matrix<T>> jac = std::make_shared<Matrix<T>>(dy->rows, dy->rows); // 输入输出大小一致，所以雅可比shape长宽一致
        std::shared_ptr<Matrix<T>> dx = std::make_shared<Matrix<T>>(dy->rows, dy->cols);
        softmaxJac<T><<<dim3(grid_y, grid_y), dim3(block_x, block_y)>>>(output->data_device.get(), jac->data_device.get(), dy->rows);
        TmulKernel<T, block_x, 1><<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(jac->data_device.get(), dy->data_device.get(), dx->data_device.get(), dy->rows, dy->cols, dy->rows);
        return dx;
    }
};