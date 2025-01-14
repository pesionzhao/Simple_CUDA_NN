#pragma once
#include <iostream>
#include <vector>
#include "../nn_utils/Matrix.h"
#include "../nn_utils/launch.h"

template<typename T>
class NNLayer {
protected:
	std::string name;
public:
	std::vector<std::shared_ptr<Matrix<T>>> params;
	bool train=true;
	virtual ~NNLayer() = 0;
	virtual std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> A) = 0;
	virtual std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dZ) = 0;
	std::string getName() { return this->name; };

};

template<typename T>
inline NNLayer<T>::~NNLayer() {}