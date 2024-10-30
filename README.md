# HW2P2: Deep Learning Model Implementation and Ablation Study

This repository contains the code for Homework 2 Part 2 in the "Intro to Deep Learning" course. This project focuses on implementing a deep learning model, specifically fine-tuning ResNet-18, to achieve the final Equal Error Rate (EER) target. The model was trained, evaluated, and optimized with a range of ablations using [Weights & Biases]([https://wandb.ai/skandv](https://api.wandb.ai/links/skandv-carnegie-mellon-university/6krkcecg)) for detailed tracking and performance insights.


## Introduction

This project involves implementing a deep learning model to perform classification, with ResNet-18 as the final architecture yielding an EER of 9.54. The project explores various configurations, aiming to optimize model performance through extensive ablations.


### Prerequisites

To run this project, you will need:

- Python 3.8 or later
- CUDA-compatible GPU (recommended)
- Required Python packages (see `requirements.txt`)



## Dataset
Download the dataset from Kaggle and place it in the data/ directory. Instructions for loading and preprocessing the data are provided in the notebook.

## Ablation Study
Extensive ablations were conducted to assess the effect of various hyperparameters:

## Model Architecture: ResNet-18 was chosen for its balance between accuracy and computational efficiency.
Learning Rate and Optimizers: Experiments were conducted with SGD and AdamW, with dynamic learning rate scheduling.
Activation Functions: GELU and ReLU were compared.
Schedulers: StepLR, ReduceLROnPlateau, and Cosine Annealing Warm Restarts were evaluated.
Batch Size and Epochs: Batch sizes and epochs were tuned to maximize convergence and generalization.
Observations
The Weights & Biases project provides insights and visualizations, revealing that:

## Architecture Choice: ResNet-18 provided the best trade-off between accuracy and computation.
## Learning Rate Scheduling: ReduceLROnPlateau improved validation accuracy.
## Optmizer: SGD Performed better than AdamW and Adam
## Activation Function: ReLU performed better.
## Refer to the Weights & Biases report for detailed graphs and comparisons.


