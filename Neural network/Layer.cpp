//
//  Layer.cpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#include "Layer.hpp"
#include <random>
#include <mutex>
#include <thread>

Layer::Layer(int numNodesIn, int numNodesOut)
: m_weights(numNodesIn * numNodesOut), m_biases(numNodesOut), m_weightGradients(m_weights.size()), m_biasGradients(m_biases.size()), m_weightVelocities(m_weights.size(), 0), m_biasVelocities(m_biases.size(), 0)
{
    m_numNodesIn = numNodesIn;
    m_numNodesOut = numNodesOut;
}

std::vector<float> Layer::CalculateOutputs(std::vector<float> inputs, ActivationType activationType)
{    
    std::vector<float> outputs(m_numNodesOut);

    for (int nodeOut = 0; nodeOut < m_numNodesOut; nodeOut++)
    {
        float weightedInput = 0;

        for (int nodeIn = 0; nodeIn < m_numNodesIn; nodeIn++)
        {
            weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
        }
        
        weightedInput += m_biases[nodeOut];
        
        outputs[nodeOut] = weightedInput;
    }
        
    for (int outputNode = 0; outputNode < m_numNodesOut; outputNode++)
    {
        outputs[outputNode] = Activation::activate(outputs, outputNode, ActivationType::Sigmoid);
    }
    
    return outputs;
}

std::vector<float> Layer::CalculateOutputs(LayerLearnData& layerData, std::vector<float> inputs, ActivationType activationType)
{
    layerData.inputs = inputs;

    for (int nodeOut = 0; nodeOut < m_numNodesOut; nodeOut++)
    {
        float weightedInput = m_biases[nodeOut];

        for (int nodeIn = 0; nodeIn < m_numNodesIn; nodeIn++)
        {
            weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
        }
        
        layerData.weightedInputs[nodeOut] = weightedInput;
    }
        
    for (int i = 0; i < layerData.activations.size(); i++)
    {
        layerData.activations[i] = Activation::activate(layerData.weightedInputs, i, ActivationType::Sigmoid);
    }
    
    return layerData.activations;
}

void Layer::CalculateOutputLayerNodeValues(LayerLearnData& layerData, std::vector<float> expectedOutputs, CostType costType)
{
    for (int i = 0; i < layerData.nodeValues.size(); i++)
    {
        float costDerivative = Cost::derivative(layerData.activations[i], expectedOutputs[i], CostType::MeanSquareError);
        float activationDerivative = Activation::derivative(layerData.weightedInputs, i, ActivationType::Softmax);
        layerData.nodeValues[i] = costDerivative * activationDerivative;
    }
}

void Layer::CalculateLayerNodeValues(LayerLearnData& layerData, Layer oldLayer, std::vector<float> oldNodeValues, ActivationType activationType)
{
    for (int newNodeIndex = 0; newNodeIndex < m_numNodesOut; newNodeIndex++)
    {
        float newNodeValue = 0;
        for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.size(); oldNodeIndex++)
        {
            // Partial derivative of the weighted input with respect to the input
            float weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
            newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
        }
        newNodeValue *= Activation::derivative(layerData.weightedInputs, newNodeIndex, activationType);
        layerData.nodeValues[newNodeIndex] = newNodeValue;
    }
}

void Layer::updateGradients(LayerLearnData& layerData)
{
    std::mutex mtx;
    mtx.lock();
    {
        for (int nodeOut = 0; nodeOut < m_numNodesOut; nodeOut++)
        {
            float nodeValue = layerData.nodeValues[nodeOut];
            for (int nodeIn = 0; nodeIn < m_numNodesIn; nodeIn++)
            {
                m_weightGradients[GetFlatWeightIndex(nodeIn, nodeOut)] += layerData.inputs[nodeIn] * nodeValue;
            }
        }
        
        for (int nodeOut = 0; nodeOut < m_numNodesOut; nodeOut++)
        {
            m_biasGradients[nodeOut] += layerData.nodeValues[nodeOut];
        }
    }
    mtx.unlock();
}

void Layer::applyGradient(float learnRate, float regularization, float momentum)
{
    float weightDecay = (1.0f - regularization * learnRate);

    for (int i = 0; i < m_weights.size(); i++)
    {
        float velocity = m_weightVelocities[i] * momentum - m_weightGradients[i] * learnRate;
        m_weightVelocities[i] = velocity;
        m_weights[i] = m_weights[i] * weightDecay + velocity;
        m_weightGradients[i] = 0;
    }

    for (int i = 0; i < m_biases.size(); i++)
    {
        float velocity = m_biasVelocities[i] * momentum - m_biasGradients[i] * learnRate;
        m_biasVelocities[i] = velocity;
        m_biases[i] += velocity;
        m_biasGradients[i] = 0;
    }
}

// Calculate layer output activations and store inputs/weightedInputs/activations in the given learnData object
//std::vector<float> Layer::CalculateOutputs(std::vector<float> inputs, LayerLearnData learnData)
//{
//    learnData.inputs = inputs;
//
//    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
//    {
//        float weightedInput = biases[nodeOut];
//        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
//        {
//            weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
//        }
//        learnData.weightedInputs[nodeOut] = weightedInput;
//    }
//
//    for (int i = 0; i < learnData.activations.size(); i++)
//    {
//        learnData.activations[i] = activation.activate(learnData.weightedInputs, i);
//    }
//
//    return learnData.activations;
//}

//void Layer::ApplyGradients(float learnRate, float regularization, float momentum)
//{
//    float weightDecay = (1 - regularization * learnRate);
//
//    for (int i = 0; i < weights.size(); i++)
//    {
//        float weight = weights[i];
//        float velocity = weightVelocities[i] * momentum - costGradientW[i] * learnRate;
//        weightVelocities[i] = velocity;
//        weights[i] = weight * weightDecay + velocity;
//        costGradientW[i] = 0;
//    }
//
//
//    for (int i = 0; i < biases.size(); i++)
//    {
//        float velocity = biasVelocities[i] * momentum - costGradientB[i] * learnRate;
//        biasVelocities[i] = velocity;
//        biases[i] += velocity;
//        costGradientB[i] = 0;
//    }
//}

float Layer::GetWeight(int nodeIn, int nodeOut)
{
    int flatIndex = nodeOut * m_numNodesIn + nodeIn;
    return m_weights[flatIndex];
}

int Layer::GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex)
{
    return outputNeuronIndex * m_numNodesIn + inputNeuronIndex;
}
//
//public void SetActivationFunction(IActivation activation)
//{
//    this.activation = activation;
//}

void Layer::initRandomWeights()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    
    for (int i = 0; i < m_weights.size(); i++)
    {
        m_weights[i] = distribution(generator);
    }
    
    for (int i = 0; i < m_biases.size(); i++)
    {
        m_biases[i] = distribution(generator);
    }
    
}
