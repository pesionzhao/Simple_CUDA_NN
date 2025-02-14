#pragma once
#include"../layers/softmax.cuh"
#include"loss.cuh"
//一维向量的交叉熵
//第一维为batchsize
template<typename T>
__global__ void CEforward(T* pre, T* tar, T* res, int M, int N){
    int cols_per_thread = (N+31)/32;
    int m_idx = blockIdx.x*blockDim.y+threadIdx.y;
    for(int row = m_idx; row<M; row+=gridDim.x*blockDim.y){
        T sum = 0;
        for(int i = 0; i<cols_per_thread; i++){
            int col = 32*i+threadIdx.x;
            if(col<N){
                T y_i = tar[row*N+col];
                T y_hat = pre[row*N+col];
                y_hat = y_hat ? y_hat:1e-5;
                sum += -y_i*log(y_hat);
            }
        }
        sum = warpReduce<SumOp, T, 32>(sum);
        if(threadIdx.x == 0) {
            res[row] = sum;
            // printf("CE sum = %f", sum);
        }
    }
}
template <typename T>
class CE : public Loss<T>{
public:
    std::shared_ptr<Tensor<T>> y_hat;
    float cost(std::shared_ptr<Tensor<T>> predictions, std::shared_ptr<Tensor<T>> target){
        //TODO 形状判断 if(pre->shape!=tar->shape) throw
        if(predictions->shape!=target->shape){
            std::ostringstream oss;
            oss << "LinearLayer error: predictions->shape != target->shape which " 
                << predictions->rows << " != " << target->rows;
            throw std::runtime_error(oss.str()); // 抛出异常并传递格式化后的错误消息
        }
        //假设输入未作softmax
        this->predictions = predictions;
        std::shared_ptr<Softmax<T>> softmax = std::make_shared<Softmax<T>>();
        this->y_hat = softmax->forward(predictions);
        this->target = target;
        int M = this->y_hat->rows;
        int N = this->y_hat->cols;
        //两个一维向量做交叉熵损失
        dim3 block_size(32,4);
        int grid_size = (block_size.y+M-1)/M;
        T* res_device = nullptr;
        cudaMalloc((void**)&res_device, M*sizeof(T));
        CEforward<T><<<grid_size, block_size>>>(this->y_hat->data_device.get(), target->data_device.get(), res_device, M, N);
        T* loss_device = nullptr;
        cudaMalloc((void**)&loss_device, sizeof(T));
        reduce<T><<<1, 32>>>(res_device, loss_device, M);
        cudaFree(res_device);
        T* res_host = new T;
        cudaMemcpy(res_host, loss_device, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(loss_device);
        std::cout<<std::scientific<< std::setprecision(8)<<"CE loss: " << *res_host/M << std::endl;
        return *res_host;
    }
    //针对带有softmax的CE, dL = y_hat - y
    std::shared_ptr<Tensor<T>> dCost(){
        std::shared_ptr<Tensor<T>> dL = std::make_shared<Tensor<T>>(this->predictions->rows, this->predictions->cols);
        *dL = *this->y_hat - *this->target;
        return dL;
    }
};