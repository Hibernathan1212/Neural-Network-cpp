//
//  main.cpp
//  Neural network
//
//  Created by Nathan Thurber on 28/6/24.
//

#include <iostream>
#include <vector>
#include <cmath>
//#include <distance>

#include "Activation.hpp"
#include "Cost.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"

int main(int argc, const char * argv[]) 
{
    Network network( {784, 100, 100, 10}, ActivationType::Sigmoid, CostType::CrossEntropy );
    
    network.initRandomWeights();
    
    network.loadData("/Users/nathan/Downloads/mnist dataset/mnist_train.csv", 784, 60000);
    
    network.train(10, 100, 1.0f, 0.1f, 0.9f);

    network.clearData();

    network.loadData("/Users/nathan/Downloads/mnist dataset/mnist_test.csv", 784, 10000);
    
    network.test();
    
    network.loadWeights("/../..");
    
    //network.saveWeights("/../..");
    //
    //network.calculateOutput( {1, 1, 1} ); //max value instead of whole vector
    //
    //network.calculateOutput("/../.."); //max value instead of whole vector
    
    return 0;
}
