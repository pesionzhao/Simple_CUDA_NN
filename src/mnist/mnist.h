#pragma once
#include<fstream>
#include<sstream>
#include<vector>
#include<iostream>
#include<string>
class Mnist{
public:
    std::vector<std::shared_ptr<Tensor<float>>> data;
    std::vector<std::shared_ptr<Tensor<float>>> label;
    int len;
    int batch_size;
    int num_batch;
    Mnist(const std::string filename){
        std::vector<std::string> lines = readCSV(filename);
        len = lines.size();
        batch_size = 10;
        num_batch = len/batch_size;
        for (int i = 0; i < len; i+=batch_size) {
            int bs = min(len-i, batch_size);
            std::shared_ptr<Tensor<float>> aData = std::make_shared<Tensor<float>>(bs, 784, true);
            std::shared_ptr<Tensor<float>> aLabel = std::make_shared<Tensor<float>>(bs, 10, false);
            aLabel->zeroInitHost();
            std::vector<int> all_values(785);
            std::string item;
            for(int batch =0; batch<bs;  batch++){
                int idx = 0;
                std::stringstream ss(lines[i+batch]);
                while (std::getline(ss, item, ',')) {
                    all_values[idx++] = std::stoi(item);
                }
                aLabel->data_host.get()[batch*10+all_values[0]] = 1.0;
                for(int j = 0; j<784; ++j){
                    aData->data_host.get()[j+784*batch] = (all_values[j+1] / 255.0f);
                    aData->data_host.get()[j+784*batch] = (aData->data_host.get()[j+784*batch] - 0.1307)/0.3081;
                }
            }
            aData->copyHostToDevice();
            aLabel->copyHostToDevice();
            data.push_back(aData); 
            label.push_back(aLabel); 
        }
    };
    std::vector<std::string> readCSV(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    }
    std::shared_ptr<Tensor<float>> getItem(int idx){
        // std::cout<<*data[idx];
        return data[idx];
    }
    std::shared_ptr<Tensor<float>> getLabel(int idx){
        // std::cout<<*label[idx];
        return label[idx];
    }
};