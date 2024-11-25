//
//  Cost.cpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#include "Cost.hpp"

#include <cmath>

float Cost::getCost(std::vector<float> outputs, std::vector<float> expectedOutputs, CostType type)
{
    if (type == CostType::MeanSquareError)
    {
        float cost = 0;
        for (int i = 0; i < outputs.size(); i++)
        {
            float error = outputs[i] - expectedOutputs[i];
            cost += error * error;
        }
        return 0.5 * cost;
    }
    else if (type == CostType::CrossEntropy)
    {
        float cost = 0;
        for (int i = 0; i < outputs.size(); i++)
        {
            float x = outputs[i];
            float y = expectedOutputs[i];
            float v = (y == 1) ? -log(x) : -log(1 - x);
            cost += v ? 0 : v;
        }
        return cost;
    }
    else
    {
        throw std::runtime_error("Invalid cost function");
    }
}

float Cost::derivative(float output, float expectedOutput, CostType type)
{
    if (type == CostType::MeanSquareError)
    {
        return output - expectedOutput;
    }
    else if (type == CostType::CrossEntropy)
    {
        float x = output;
        float y = expectedOutput;
        if (x == 0 || x == 1)
        {
            return 0;
        }
        return (-x + y) / (x * (x - 1));

    }
    else
    {
        throw std::runtime_error("Invalid cost function");
    }
}
