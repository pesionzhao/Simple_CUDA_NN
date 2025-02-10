#pragma once

#include <string>
#include <vector>
#include "../nn_utils/Matrix.h"

template<typename T>
class Loss{
protected:
    std::string name;
    std::shared_ptr<Matrix<T>> predictions;
    std::shared_ptr<Matrix<T>> target;
public:
    virtual float cost(std::shared_ptr<Matrix<T>> predictions, std::shared_ptr<Matrix<T>> target) = 0;
    virtual std::shared_ptr<Matrix<T>> dCost() = 0;
    const std::string getName() const { return this->name; };
    void backwardPass() {
        std::vector<std::shared_ptr<Matrix<T>>> topo;
        //为什么要构建visited,不能存在多条路径吗
        std::unordered_set<std::shared_ptr<Matrix<T>>> visited;
        //DFS构建计算图
        std::function<void(std::shared_ptr<Matrix<T>>)> buildTopo = [&topo, &visited, &buildTopo](std::shared_ptr<Matrix<T>> v) {
            // std::cout<<"node op is "<< v->op << " and prev.size() = " <<v->prev.size() << std::endl;
            if (visited.insert(v).second) {
                for (auto child : v->prev) {
                    buildTopo(child);
                }
                topo.push_back(v);
                std::cout<<"push node "<< v->op << std::endl;
            }
        };

        buildTopo(this->predictions);
        std::cout<<"topo size is "<< topo.size()<<std::endl;
        this->predictions->grad = dCost(); // seed the gradient、
        //要按照dfs遍历的顺序进行反向传播 以保证在进行梯度反向传播时，所有依赖的节点的梯度都已经计算好了。 
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->grad_fn(*it);//计算所有节点的梯度
        }
    }
};