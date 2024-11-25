//
//  Cost.hpp
//  Neural network
//
//  Created by Nathan Thurber on 8/7/24.
//

#pragma once

#include <vector>

enum class CostType
{
    MeanSquareError,
    CrossEntropy
};

class Cost
{
public:
    static float getCost(std::vector<float> outputs, std::vector<float> expectedOutputs, CostType type = CostType::MeanSquareError);
    
    static float derivative(float output, float expectedOutput, CostType type = CostType::MeanSquareError);
    
};
