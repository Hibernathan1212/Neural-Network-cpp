//
//  NeuralNetwork.cpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#include "NeuralNetwork.hpp"
#include "Layer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <thread>

#include <future>

Network::Network(std::vector<int> layerSizes, ActivationType activationType, CostType costType)
: m_activationType(activationType), m_costType(costType)
{
    m_layerSizes = layerSizes;

    for (int i = 0; i < m_layerSizes.size() - 1; i++)
    {
        m_layers.push_back(Layer(m_layerSizes[i], m_layerSizes[i + 1]));
    }
}

float Network::test()
{
    float accuracy = 0;
    int numCorrect = 0;
    
    for (int i = 0; i < m_dataSize; i++)
    {
        std::vector<float> inputs(m_data[i].begin() + 1, m_data[i].end());
        int output = maxValueIndex(forwardPass(inputs));
        if (output == m_data[i][0])
        {
            numCorrect += 1;
        }
    }
    
    accuracy = numCorrect / (float) m_dataSize;
    
    std::cout << numCorrect << "/" << m_dataSize << std::endl;
    
    std::cout << accuracy * 100.0f << "%" << std::endl;
    
    return accuracy;
}

void Network::train(int iterations, int miniBatchSize, float learnRate, float regularization, float momentum)
{
    if (m_data.size() == 0)
    {
        std::cerr << "[Network train] Invalid dataset";
        return;
    }
        
    for (int iteration = 0; iteration < iterations; iteration++)
    {
        learnRate *= 0.8f;
        for (int batch = 0; batch < m_data.size(); batch += miniBatchSize)
        {
            m_numCorrect = 0;
            
            std::vector<std::thread> thread_group;
            thread_group.reserve(miniBatchSize);
            
            for (int i = 0; i < miniBatchSize; i++)
            {
                //std::future<void> EventualValue = std::async(std::launch::async, [this, &learnData, i, batch](){ this->updateGradients(learnData, batch + i); });
                thread_group.emplace_back([this, i, batch](){ this->updateGradients(batch + i); });
                //updateGradients(batch + i);
            }
            
            for (auto& thread : thread_group)
            {
                thread.join();
            }
            
            for (int j = 0; j < m_layers.size(); j++)
            {
                m_layers[j].applyGradient(learnRate / miniBatchSize, regularization, momentum);
            }
            
            float accuracy = m_numCorrect / (float) miniBatchSize;
            
            //std::cout << numCorrect << "/" << miniBatchSize << std::endl;
            
            std::cout << "[Epoch: " << iteration << "] " << accuracy * 100.0f << "%" << std::endl;
        }
    }
}

void Network::updateGradients(int batch)
{
    std::vector<LayerLearnData> learnData(m_layers.size());
    
    for (int layer = 0; layer < m_layers.size(); layer++)
    {
        learnData[layer].inputs.resize(m_layerSizes[layer]);
        learnData[layer].weightedInputs.resize(m_layerSizes[layer + 1]);
        learnData[layer].activations.resize(m_layerSizes[layer + 1]);
        learnData[layer].nodeValues.resize(m_layerSizes[layer + 1]);
    }
    
    std::mutex mtx;
    
    //std::vector<float> data = m_data[batch];
    
    std::vector<float> inputs(m_data[batch].begin() + 1, m_data[batch].end());
    std::vector<float> outputs = forwardPass(inputs, learnData);
    
    // -- Backpropagation --
    // Update output layer gradients
    std::vector<float> expectedOutputs(m_layerSizes[m_layerSizes.size() - 1]);
    expectedOutputs[m_data[batch][0]] = 1.0f;
    
    m_layers[m_layers.size() - 1].CalculateOutputLayerNodeValues(learnData[learnData.size() - 1], expectedOutputs, m_costType);
    m_layers[m_layers.size() - 1].updateGradients(learnData[learnData.size() - 1]);
    
    // Update all hidden layer gradients
    for (int index = m_layers.size() - 2; index >= 0; index--)
    {
        m_layers[index].CalculateLayerNodeValues(learnData[index], m_layers[index + 1], learnData[index + 1].nodeValues,    m_activationType);
        m_layers[index].updateGradients(learnData[index]);
    }
    
    if (maxValueIndex(outputs) == m_data[batch][0])
    {
        mtx.lock();
        m_numCorrect += 1;
        mtx.unlock();
    }
}

std::vector<float> Network::forwardPass(std::vector<float> inputs)
{
    for (int i = 0; i < m_layers.size() - 1; i++)
    {
        inputs = m_layers[i].CalculateOutputs(inputs, m_activationType);
    }
    
    std::vector<float> output = m_layers[m_layers.size() - 1].CalculateOutputs(inputs, ActivationType::Softmax);
    
    return output;
}

std::vector<float> Network::forwardPass(std::vector<float> inputs, std::vector<LayerLearnData>& layerData)
{
    for (int i = 0; i < m_layers.size() - 1; i++)
    {
        inputs = m_layers[i].CalculateOutputs(layerData[i], inputs, m_activationType);
    }
    
    std::vector<float> output = m_layers[m_layers.size() - 1].CalculateOutputs(layerData[m_layers.size() - 1], inputs, ActivationType::Softmax);
    
    return output;
}

void Network::initRandomWeights()
{
    for (Layer& layer : m_layers)
    {
        layer.initRandomWeights();
    }
}

void Network::loadWeights(std::string filePath)
{
    
}

void saveWeights(std::string filePath)
{
    
}

void Network::loadData(std::string filePath, int numInputs, int dataSize)
{
    m_dataSize = dataSize;
    m_numInputs = numInputs;
    
    m_data.resize(m_dataSize);
    
    if (m_numInputs != m_layerSizes[0])
    {
        std::cerr << "Input layer resized to " << m_numInputs << " from " << m_layerSizes[0] << std::endl;
        
        m_layers.clear();
        m_layers.push_back(Layer(m_numInputs, m_layerSizes[1]));
        for (int i = 1; i < m_layers.size(); i++)
        {
            m_layers.push_back(Layer(m_layerSizes[i], m_layerSizes[i + 1]));
        }

        m_layerSizes[0] = m_numInputs;
    }
    
    std::string contents;
    {
        std::fstream file(filePath, std::ios::in);
        std::stringstream contents_stream;
        contents_stream << file.rdbuf();
        file.close();
        contents = contents_stream.str();
    }
    
    size_t index = 0;
    int i = 0;
    int input = 0;
    std::string buf;
    
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.001f);
    
    while (index < contents.length() && i <= m_dataSize)
    {
        if (contents.at(index) == ',' || contents.at(index) == '\r')
        {
            if (i > 0)
            {
                if (input != 0)
                    m_data[i - 1].push_back(distribution(generator) + std::stof(buf) / 255.0f); //noise
                else
                    m_data[i - 1].push_back(std::stof(buf));

            }
            
            buf.clear();
            input++;
            if (input == m_numInputs + 1)
            {
                input = 0;
                i++;
            }
        }
        else
        {
            buf.push_back(contents.at(index));
        }
        index++;
    }
}

void Network::clearData()
{
    m_data.clear();
    m_dataSize = 0;
}

int Network::maxValueIndex(std::vector<float> values)
{
    float maxValue = values[0];
    int index = 0;
    for (int i = 1; i < values.size(); i++)
    {
        if (values[i] > maxValue)
        {
            maxValue = values[i];
            index = i;
        }
    }

    return index;
}
