//
//  Layer.hpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#pragma once

#include <vector>

#include "Activation.hpp"
#include "Cost.hpp"

struct LayerLearnData
{
    std::vector<float> inputs;
    std::vector<float> weightedInputs;
    std::vector<float> activations;
    std::vector<float> nodeValues;
};

class Layer
{
public:
    Layer(int numNodesIn, int numNodesOut);
    
    std::vector<float> CalculateOutputs(std::vector<float> inputs, ActivationType activationType);
    
    std::vector<float> CalculateOutputs(LayerLearnData& layerData, std::vector<float> inputs, ActivationType activationType);

    void CalculateOutputLayerNodeValues(LayerLearnData& layerData, std::vector<float> expectedOutputs, CostType costType);
        
    void CalculateLayerNodeValues(LayerLearnData& layerData, Layer oldLayer, std::vector<float> oldNodeValues, ActivationType activationType);

    void updateGradients(LayerLearnData& layerData);
    
    void applyGradient(float learnRate, float regularization, float momentum);
    
    //std::vector<float> CalculateOutputs(std::vector<float> inputs, LayerLearnData learnData);
    
    //void ApplyGradients(float learnRate, float regularization, float momentum);
    
    //void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, std::vector<float> expectedOutputs);
    
    //void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer, std::vector<float> oldNodeValues);
    
    
    void initRandomWeights();
        
private:
    
    int m_numNodesIn;
    int m_numNodesOut;

    std::vector<float> m_weights;
    std::vector<float> m_biases;

    std::vector<float> m_weightGradients;
    std::vector<float> m_biasGradients;

    std::vector<float> m_weightVelocities;
    std::vector<float> m_biasVelocities;
          
    //Activation m_activation;
    //Cost m_cost;
    
private:
    float GetWeight(int nodeIn, int nodeOut);
    int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex);
};
