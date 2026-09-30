//
//  Activation.hpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#pragma once

#include <vector>

enum class ActivationType
{
    Sigmoid,
    TanH,
    ReLU,
    SiLU,
    Softmax
};

class Activation
{
public:
    static float activate(std::vector<float> inputs, int index, ActivationType type = ActivationType::Sigmoid);

    static float derivative(std::vector<float> inputs, int index, ActivationType type = ActivationType::Sigmoid);
};
