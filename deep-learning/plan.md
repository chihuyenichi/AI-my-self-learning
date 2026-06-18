## Deep Learning Fundamentals

### Overview
Deep learning is an optimization process that learns patterns from data by adjusting internal parameters (weights) to minimize prediction error.

### Core Process Flow
1. **Forward Pass**: Input X passes through layers → Produces prediction Y'
2. **Loss Calculation**: Compare Y' with actual output Y using a Loss Function
3. **Optimization**: Compute loss score and adjust weights to minimize error
4. **Backpropagation**: Repeat the process with updated weights

### Key Concepts
- **Loss Function**: Measures quality of network's output; goal is to minimize this value
- **Weight Updates**: Network learns by iteratively adjusting parameters based on loss
- **Iterative Learning**: Process repeats until convergence or satisfactory performance


## Neural Networks

### Summary
A neural network learns by example, passing data through layers of mathematical transformations to make predictions or classifications. The network structure allows it to learn complex patterns through weight adjustment during training.

### Architecture Components
1. **Input Layer**: Receives the initial data
2. **Hidden Layers**: Process data through complex mathematical functions
3. **Output Layer**: Produces predictions or classifications

### Training Process
- Network adjusts **weights** during training to maximize accuracy
- Each layer transforms data using learned parameters
- Hidden layers extract increasingly abstract features from raw input


![Neural Network Training Workflow](/image-materials/network-trainning-work-flow.png)

## Kernel Methods

### Summary
Kernel methods map original data into high-dimensional space where it becomes linearly separable. They use a mathematical shortcut called the "**Kernel Trick**" to efficiently compute similarity without explicit transformation.

### Key Features
- **Kernel Trick**: Computes similarity between data points in high-dimensional space without explicitly transforming the data
- **Bridging Linear and Non-Linear**: Enables linear classifiers (like SVMs) to learn complex non-linear decision boundaries
- **Computational Efficiency**: Complexity depends on number of data samples, not dimensionality

### Applications
- Support Vector Machines (SVMs) with non-linear kernels
- Efficient classification in high-dimensional spaces
- Pattern recognition with implicit feature spaces

