#pragma once
#include <iostream>
#include <vector>
#include "../nn_utils/tensor.h"
#include "../nn_utils/launch.h"
#include"../nn_utils/rand.h"
template<typename T>
class NNLayer {
// protected:
public:
	std::string name;
	std::vector<std::shared_ptr<Tensor<T>>> params;
	bool train=true;
	virtual ~NNLayer() = 0;
	virtual std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> A) = 0;
	std::string getName() { return this->name; };

};

template<typename T>
inline NNLayer<T>::~NNLayer() {}