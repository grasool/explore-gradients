# Exploring the Problem of Exploding and Vanishing Gradients in Deep Neural Netwroks
Increasing the depth of a neural netwrok generally leads to increased accuracy. However, with the increasing number of layers in a neural netwroks, the gradients of the loss function with respect to the unknown parameters (weights and biases) may either explode or vanish.

I am using MNIST dataset and MLPs to explore this problem.

## Requirements
Keras 2.1.5

TensorFlow 1.5.0

Python 3.6.4 Anaconda

## Results

#### MLP with Batch Normalization and ReLU
The effect of increasing number of layers from 10 to 40
![picture2](https://user-images.githubusercontent.com/15803477/38772451-f84e0288-4004-11e8-885e-bf7d9b37aef4.png)

#### MLP without Batch Normalization and ReLU
![picture2](https://user-images.githubusercontent.com/15803477/38772509-6c6bb0e2-4006-11e8-80e1-071aece52b0a.png)

#### MLP with Skip Connections ReLU
![picture3](https://user-images.githubusercontent.com/15803477/38772511-6fe84fbe-4006-11e8-954f-ff9aea5f976c.png)

#### Maximum validation accuracy after 20 epochs
