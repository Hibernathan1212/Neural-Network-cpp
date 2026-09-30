# Neural network

A fast, from-scratch C++ feedforward neural network that reaches 90%+ accuracy on MNIST using mini-batch training, multithreaded backpropagation, and configurable activations/losses.

## Overview
- Configurable multi-layer perceptron (MLP) with arbitrary layer sizes
- Multiple activation functions: Sigmoid, SiLU/Swish, Softmax (output) [extensible]
- Multiple loss functions: Mean Squared Error (MSE), Cross Entropy
- Mini-batch SGD with momentum and L2 regularization
- Multithreaded gradient computation per mini-batch
- Proven on MNIST: 85%+ accuracy with a simple 784-100-10 architecture

## Results
- Dataset: MNIST (60k train / 10k test), CSV format
- Model: 784-100-10, activation=SiLU, loss=CrossEntropy
- Training: 10 epochs, batch size 100, lr 1.0 → decays per epoch, momentum 0.9, L2 0.1
- Test accuracy: 85–90% (typical)

Notes:
- Accuracy depends on initialization and hyperparameters; try ReLU/GELU and He/Xavier init for further gains.
- Consider shuffling and a learning-rate schedule for improved convergence.

