#pragma once
#include"optim.cuh"
template<typename T>
class Adam : public Optimizer<T>{
private:
    std::vector<std::shared_ptr<Matrix<double>>> exp_avg;
    std::vector<std::shared_ptr<Matrix<double>>> exp_avg_sq;
    double beta1, beta2, eps;
    int t=0;
public:
    Adam(std::vector<NNLayer<T>*> layers, float lr, double beta1=0.9, double beta2=0.99, double eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){
        this->lr = lr;
        for (NNLayer<T>* layer : layers) {
            for (std::shared_ptr<Matrix<T>> p : layer->params) {
                this->parameters.push_back(p);
                exp_avg.push_back(std::make_shared<Matrix<double>>(p->rows, p->cols, false));
                exp_avg_sq.push_back(std::make_shared<Matrix<double>>(p->rows, p->cols, false));
            }
        }

    }
    void step(){
        for(int i = 0; i< this->parameters.size(); i++){
            *exp_avg[i] = *exp_avg[i] * beta1 +  *this->parameters[i]->grad * (1 - beta1);
            *exp_avg_sq[i] = *exp_avg_sq[i] * beta2 +  *this->parameters[i]->grad * *this->parameters[i]->grad * (1 - beta2);
            Matrix<double> exp_avg_hat = *exp_avg[i] /  (1- pow(beta1, t+1));
            Matrix<double> exp_avg_sq_hat = *exp_avg_sq[i] /  (1- pow(beta2, t+1));
            Matrix<double> update = exp_avg_hat * (-this->lr) / (exp_avg_sq_hat.sqrt_() + eps);
            *this->parameters[i] = *this->parameters[i] + update;
        }
        t++;
        // for (auto* p : model.parameters()) {
        //     // 梯度更新逻辑
        //     p->m = beta1 * p->m + (1 - beta1) * p->grad;
        //     p->v = beta2 * p->v + (1 - beta2) * p->grad * p->grad;
        //     double m_hat = p->m / (1 - pow(beta1, step + 1));  // 偏差修正
        //     double v_hat = p->v / (1 - pow(beta2, step + 1));
        //     p->data -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p->data);
        // }
    }
    void zero_grad() {
        for (std::shared_ptr<Matrix<T>> p : this->parameters) {
            if (p->grad != nullptr) {
                p->grad->zeroInitDevice();
            }
        }
    }
    
};