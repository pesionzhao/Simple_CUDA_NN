#pragma once
#include "Matrix.h"

/*================launch basical kernel================*/
//launch add kernel
template<typename T>
std::shared_ptr<Matrix<T>> add_(Matrix<T>* src1, Matrix<T>* src2, bool requires_grad){
    if (!src1 || !src2) {
        throw std::invalid_argument("error occurs in add_, Cannot add nullptr ptr");
    }
    //src1 + src2
    if(src1->shape!=src2->shape)
        throw NNException("error occurs in add_,  + operator get the different shapes");
    int rows = src1->rows;
    int cols = src1->cols;
    std::shared_ptr<Matrix<T>> dst = std::make_shared<Matrix<T>>(rows, cols, requires_grad);
    dim3 block_size(16,16);
    dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
    add<T,T><<<grid_size, block_size>>>(src1->data_device.get(), src2->data_device.get(), dst->data_device.get(), rows, cols);
    NNException::throwIfDeviceErrorsOccurred("matmul error\n");
    return dst;
};
//src1 @ src2 
//src1->cols == src2->rows
template<typename T>
std::shared_ptr<Matrix<T>> mul_(Matrix<T>* src1, Matrix<T>* src2){
    const int M = src1->rows;
    const int N = src2->cols;
    const int K = src1->cols;
    std::shared_ptr<Matrix<T>> dst = std::make_shared<Matrix<T>>(M, N, src1->requires_grad||src2->requires_grad);
    dim3 block_size(16,16);
    dim3 grid_size((N + block_size.x - 1)/ block_size.x, (M + block_size.y - 1)/ block_size.y);
    mulKernel<T, 16, 1><<<grid_size, block_size>>>(src1->data_device.get(), src2->data_device.get(), dst->data_device.get(), M, N, K);
    NNException::throwIfDeviceErrorsOccurred("matmul error\n");
    return dst;
};
//src1.T @ src2
//src1->rows == src2->rows
template<typename T>
std::shared_ptr<Matrix<T>> Tmul_(Matrix<T>* src1, Matrix<T>* src2){
    const int M = src1->cols;
    const int N = src2->cols;
    const int K = src1->rows;
    std::shared_ptr<Matrix<T>> dst = std::make_shared<Matrix<T>>(M, N, src1->requires_grad||src2->requires_grad);
    dim3 block_size(16,16);
    dim3 grid_size((N + block_size.x - 1)/ block_size.x, (M + block_size.y - 1)/ block_size.y);
    TmulKernel<T, 16, 1><<<grid_size, block_size>>>(src1->data_device.get(), src2->data_device.get(), dst->data_device.get(), M, N, K);
    NNException::throwIfDeviceErrorsOccurred("Tmatmul error\n");
    return dst;
};
//src1 @ src2.T
//src1->cols = src2->cols
template<typename T>
std::shared_ptr<Matrix<T>> mulT_(Matrix<T>* src1, Matrix<T>* src2){
    const int M = src1->rows;
    const int N = src2->rows;
    const int K = src1->cols;
    std::shared_ptr<Matrix<T>> dst = std::make_shared<Matrix<T>>(M, N, src1->requires_grad||src2->requires_grad);
    dim3 block_size(16,16);
    dim3 grid_size((N + block_size.x - 1)/ block_size.x, (M + block_size.y - 1)/ block_size.y);
    mulTKernel<T, 16, 1><<<grid_size, block_size>>>(src1->data_device.get(), src2->data_device.get(), dst->data_device.get(), M, N, K);
    NNException::throwIfDeviceErrorsOccurred("matmulT error\n");
    return dst;
};

/*================重载智能指针运算符================*/
//std::shared_ptr<Matrix<T>> + std::shared_ptr<Matrix<T>> src2
template<typename T>
std::shared_ptr<Matrix<T>> operator+(std::shared_ptr<Matrix<T>> src1, std::shared_ptr<Matrix<T>> src2){
    bool requires_grad = src1->requires_grad||src2->requires_grad;
    std::shared_ptr<Matrix<T>> dst = add_(src1.get(), src2.get(), requires_grad);
    if(requires_grad){
        dst->op = "+";
        dst->prev.insert(src1);
        dst->prev.insert(src2);
        dst->setBackward([src1,   src2](std::shared_ptr<Matrix<T>> dst) {
            src1->addGrad(dst->grad);
            src2->addGrad(dst->grad);
        });
    }
    return dst;
}

template<typename T>
std::shared_ptr<Matrix<T>> matmul(std::shared_ptr<Matrix<T>> src1, std::shared_ptr<Matrix<T>> src2) {
    //this @ others
    std::shared_ptr<Matrix<T>> dst = mul_(src1.get(), src2.get());
    bool requires_grad = src1->requires_grad||src2->requires_grad;
    if(requires_grad){
        dst->op = "matmul";
        //更新前驱节点的梯度
        dst->prev.insert(src1);
        dst->prev.insert(src2);
        dst->setBackward([src1,  src2](std::shared_ptr<Matrix<T>> dst) {
            //对this的偏导数为
            src1->addGrad(mulT_(dst->grad.get(), src2.get()));
            //对others的偏导数为this
            //jac = *this
            // other->addGrad((this->T_).matmul(dst->grad));
            src2->addGrad(Tmul_(src1.get(), dst->grad.get()));
        });
    }
    return dst;
};