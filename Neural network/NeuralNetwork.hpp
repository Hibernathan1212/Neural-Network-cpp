//
//  NeuralNetwork.hpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#pragma once

#include "Layer.hpp"
#include <vector>

class Network
{
public:
    Network(std::vector<int> layerSizes, ActivationType activationType = ActivationType::Sigmoid, CostType costType = CostType::MeanSquareError);
    
    void initRandomWeights();
    
    void loadWeights(std::string filePath);
    
    //void saveWeights(std::string filePath);
    
    void loadData(std::string filePath, int numInputs, int dataSize);
    void clearData();
    
    void train(int iterations, int miniBatchSize = MAXFLOAT, float learnRate = 0.2f, float regularization = 0.0f, float momentum = 0.0f);
    
    void updateGradients(int batch);
    
    float test();
    
    //void learn(std::vector<std::vector<float>> trainingData, float learnRate = 0.2, float regularization = 0, float momentum = 0);
    
    //void updateGradients(std::vector<float> data, std::vector<float> expectedOutputs, NetworkLearnData learnData);
    
    std::vector<float> forwardPass(std::vector<float> inputs);
    
    std::vector<float> forwardPass(std::vector<float> inputs, std::vector<LayerLearnData>& layerData);

    
    void backwardsPass(std::vector<float> outputs);
    
    //std::vector<NetworkLearnData> learnData;
    
private:
    int maxValueIndex(std::vector<float> values);
    
private:
    std::vector<Layer> m_layers;
    std::vector<int> m_layerSizes;
    
    std::vector<std::vector<float>> m_data;
    
    int m_numInputs;
    int m_dataSize;
    
    int m_numCorrect;
    
    //first element is label/correct answer, rest is data
    
    ActivationType m_activationType;
    
    CostType m_costType;
};




//class NetworkLearnData
//{
//public:
//    NetworkLearnData(std::vector<Layer> layers);
//
//    std::vector<LayerLearnData> layerData;
//};
