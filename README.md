# Exploring the Problem of Exploding and Vanishing Gradients in Deep Neural Netwroks
Increasing the depth of a neural netwrok generally leads to increased accuracy. However, with the increasing number of layers in a neural netwroks, the gradients of the loss function with respect to the unknown parameters (weights and biases) may either explode or vanish.

I am using MNIST dataset and MLPs to explore this problem.

## Requirements
Keras 2.1.5

TensorFlow 1.5.0

Python 3.6.4 Anaconda


#### MLP with Batch Normalization and ReLU nonlinearity

