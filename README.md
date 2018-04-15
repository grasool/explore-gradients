# Exploring the Problem of Exploding and Vanishing Gradients in Deep Neural Netwroks
Increasing the depth of a neural netwrok generally leads to increased accuracy. However, with the increasing number of layers in a neural netwroks, the gradients of the loss function with respect to the unknown parameters (weights and biases) may either explode or vanish.

## Related Work
Here are some recent papers that explore the problem of exploding or vanishing gradients or both.
1.  The exploding gradient problem demystified - definition, prevalence, impact, origin, tradeoffs, and solutions (https://arxiv.org/abs/1712.05577)

2.  Mean Field Residual Networks: On the Edge of Chaos (https://arxiv.org/abs/1712.08969)

3.  Deep Mean Field Theory: Layerwise Variance and Width Variation as Methods to Control Gradient Explosion (https://openreview.net/pdf?id=rJGY8GbR-)

4.  Deep Information Propagation (https://arxiv.org/abs/1611.01232)

5.  Exponential expressivity in deep neural networks through transient chaos (https://arxiv.org/abs/1606.05340v1)

6.  Stable Architectures for Deep Neural Networks (https://arxiv.org/abs/1705.03341)

7.  Learning across scales - A multiscale method for Convolution Neural Networks (https://arxiv.org/abs/1703.02009)

8. Reversible Architectures for Arbitrarily Deep Residual Neural Networks (https://arxiv.org/abs/1709.03698) 

9.  Multi-level Residual Networks from Dynamical Systems View (2018) (https://openreview.net/pdf?id=SyJS-OgR-)

10. The Shattered Gradients Problem: If resnets are the answer, then what is the question? (https://arxiv.org/pdf/1702.08591.pdf)

11. On the Expressive Power of Deep Neural Networks (https://arxiv.org/abs/1606.05336)

12. Unitary Evolution Recurrent Neural Networks (https://arxiv.org/pdf/1511.06464.pdf)

13. Recent Advances in Recurrent Neural Networks (https://arxiv.org/pdf/1801.01078.pdf)

## Experiments
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

![11-06-2017_heimcombined2](https://user-images.githubusercontent.com/15803477/38772538-eefcf70a-4006-11e8-9411-f00ac8224a2a.png)
