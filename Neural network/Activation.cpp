//
//  Activation.cpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#include "Activation.hpp"

#include <cmath>


float Activation::activate(std::vector<float> inputs, int index, ActivationType type)
{
    switch (type)
    {
        case ActivationType::Sigmoid:
        {
            return 1.0 / (1 + exp(-inputs[index]));
        }
        case ActivationType::TanH:
        {
            float e2 = exp(2 * inputs[index]);
            return (e2 - 1) / (e2 + 1);
        }
        case ActivationType::ReLU:
        {
            return fmax(0, inputs[index]);
        }
        case ActivationType::SiLU:
        {
            return inputs[index] / (1 + exp(-inputs[index]));
        }
        case ActivationType::Softmax:
        {
            float expSum = 0;
            for (int i = 0; i < inputs.size(); i++)
            {
                expSum += exp(inputs[i]);
            }

            float res = exp(inputs[index]) / expSum;

            return res;
        }
    }
}

float Activation::derivative(std::vector<float> inputs, int index, ActivationType type)
{
    switch (type)
    {
        case ActivationType::Sigmoid:
        {
            float a = activate(inputs, index);
            return a * (1 - a);
        }
        case ActivationType::TanH:
        {
            float e2 = exp(2 * inputs[index]);
            float t = (e2 - 1) / (e2 + 1);
            return 1 - t * t;
        }
        case ActivationType::ReLU:
        {
            return (inputs[index] > 0) ? 1 : 0;
        }
        case ActivationType::SiLU:
        {
            float sig = 1 / (1 + exp(-inputs[index]));
            return inputs[index] * sig * (1 - sig) + sig;
        }
        case ActivationType::Softmax:
        {
            float expSum = 0;
            for (int i = 0; i < inputs.size(); i++)
            {
                expSum += exp(inputs[i]);
            }

            float ex = exp(inputs[index]);

            return (ex * expSum - ex * ex) / (expSum * expSum);
        }
    }
}
