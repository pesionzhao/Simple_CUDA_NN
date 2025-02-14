#pragma once

#include <string>
#include <vector>
#include "../nn_utils/tensor.h"

template<typename T>
class Loss{
protected:
    std::string name;
    std::shared_ptr<Tensor<T>> predictions;
    std::shared_ptr<Tensor<T>> target;
public:
    virtual float cost(std::shared_ptr<Tensor<T>> predictions, std::shared_ptr<Tensor<T>> target) = 0;
    virtual std::shared_ptr<Tensor<T>> dCost() = 0;
    const std::string getName() const { return this->name; };
    void backwardPass() {
        std::vector<std::shared_ptr<Tensor<T>>> topo;
        //为什么要构建visited,不能存在多条路径吗
        std::unordered_set<std::shared_ptr<Tensor<T>>> visited;
        //DFS构建计算图
        std::function<void(std::shared_ptr<Tensor<T>>)> buildTopo = [&topo, &visited, &buildTopo](std::shared_ptr<Tensor<T>> v) {
            if (visited.insert(v).second) {
                for (auto child : v->prev) {
                    buildTopo(child);
                }
                topo.push_back(v);
                // std::cout<<"push node "<< v->op << std::endl;
            }
        };

        buildTopo(this->predictions);
        // std::cout<<"topo size is "<< topo.size()<<std::endl;
        this->predictions->grad = dCost();
        //要按照dfs遍历的顺序进行反向传播 以保证在进行梯度反向传播时，所有依赖的节点的梯度都已经计算好了。 
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->grad_fn(*it);//计算所有节点的梯度
        }
    }
};