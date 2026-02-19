# Deep Learning Pipeline in PyTorch
## Overview

The goal of this project was to build a complete and reproducible Deep Learning pipeline in PyTorch. The pipeline handles:

- Dataset loading

- Model architecture

- Training

- Evaluation

- Experiment tracking

- Dataset versioning using DVC

The entire workflow is executed through main.py to ensure reproducibility and modular structure.

# Project Structure
    .
    ├── main.py
    ├── results.md
    ├── data.dvc
    ├── README.md
    └── src/
        ├── datasets.py
        ├── model.py
        ├── train.py
        ├── logger.py
        └── utils.py


# Module Description

- datasets.py - Handles CIFAR-10 dataset and DataLoader creation

- model.py - Contains the neural network architecture

- train.py - Implements the training loop and evaluation logic

- experiment_logger.py - Automatically logs experiment results

- main.py - Runs experiments and orchestrates the full pipeline

- data.dvc - Dataset version tracking

# Dataset

This project uses the CIFAR-10 dataset (32x32 RGB images across 10 classes).

The dataset is versioned using DVC in order to:

- Avoid committing raw data to Git

- Ensure reproducibility

- Keep data and code versioning separate

The data directory is managed through data.dvc, and raw data is not stored in the Git repository.

# Model Architecture

### The model is a fully connected neural network consisting of:

- Flatten layer

- Multiple Linear layers

- Batch Normalization

- LeakyReLU activations

- Dropout layers

### The architecture was modified compared to the baseline by:

- Increasing the number of hidden layers

- Adding BatchNorm

- Adding Dropout

- Replacing ReLU with LeakyReLU

These changes were made to improve generalization and reduce overfitting.

# Experiments

### Three experiments were conducted using different learning rates:

- Learning rate = 1e-3

- Learning rate = 5e-4

- Learning rate = 1e-4

Batch size and number of epochs were kept constant.

All experiments are executed automatically via:

    uv run main.py


The results are logged automatically in:

    results.md


Full experiment results can be found here:

See: [Results](./results.md)

# Reproducibility

The entire project can be reproduced using:

- dvc pull
- uv run main.py


This will:

- Pull the dataset using DVC

- Run all experiments

- Train the model

- Log results automatically

# Challenges and Learning Process

During the development of this project, several conceptual and structural challenges arose.

### Understanding logits vs probabilities

I initially struggled with understanding the difference between logits and probabilities, and how CrossEntropyLoss internally applies softmax. Clarifying this helped me better understand how the loss function operates.

### Understanding gradients and backpropagation

It took time to fully understand how loss.backward() computes gradients and how optimizer.step() updates model parameters. Breaking down the training loop step by step helped clarify how gradients propagate through the network.

### Difference between model.train() and model.eval()

Understanding the behavioral differences of BatchNorm and Dropout in training vs evaluation mode was an important realization for correctly evaluating the model.

### Using DVC

Learning how DVC separates data versioning from code versioning was an important part of the assignment. It clarified why raw data should not be stored in Git and how reproducibility is maintained in larger projects.

### Conclusion

This project resulted in a modular and reproducible Deep Learning pipeline built with PyTorch. It includes:

- Clean project structure

- Automated experiment tracking

- Dataset versioning with DVC

- Reproducible execution via a single entry point

The structure reflects how machine learning projects are organized in real-world environments.

Running the Project:

    dvc pull
    uv run main.py