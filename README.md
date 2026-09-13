# Building, Breaking and Fixing a Neural Network

## Overview

This project investigates the behaviour of a feedforward neural network using the Fashion-MNIST dataset.

The aim is to build a neural network from scratch, compare different design choices, deliberately create overfitting, and then use regularisation and hyperparameter tuning to improve generalisation.

The project is divided into seven parts:

1. Environment Setup
2. Backpropagation From Scratch
3. Baseline Model and Activation Study
4. Loss Functions
5. Optimiser Comparison
6. Forcing Overfitting
7. Regularisation Study
8. Hyperparameter Tuning with k-Fold Cross-Validation



## Dataset

The project uses the Fashion-MNIST dataset.

Dataset source:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

Fashion-MNIST contains 28 × 28 grayscale images belonging to 10 clothing classes.

Each image is:

- Normalised from pixel values of 0–255 to values between 0 and 1
- Flattened into a 784-dimensional vector

The original training data is divided using an 80/20 split:

- 80% training data
- 20% validation data

The provided test set is kept completely separate and is only used for the final evaluation in Part 7.

### Classes

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Environment

The experiments were carried out using:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Jupyter Notebook
- Kaggle

The intended Kaggle hardware configuration is:

- GPU: NVIDIA T4 × 2

If the T4 × 2 accelerator is not available, the notebook can also be run using available CPU/GPU resources, although training times may differ.
Part 1 - Backpropagation From Scratch

A two-layer multilayer perceptron was implemented using NumPy only.

No autograd or deep learning framework was used for the NumPy implementation.

# Architecture
784 input features
        |
        v
64 hidden units
        |
       ReLU
        |
        v
10 output units
        |
     Softmax

Categorical cross-entropy was used as the loss function.

The following were implemented manually:

Forward pass
ReLU activation
Softmax
Categorical cross-entropy
Backward pass
Gradient calculation
Manual gradient descent update

The network was trained on a subset of 5,000 samples for 20 epochs.

The training loss was plotted to show how the model learned during training.

## Gradient Verification

The same network was rebuilt in PyTorch.

The NumPy and PyTorch networks used:

The same initial weights
The same input batch
The same labels

One forward pass and one backward pass were performed.

The maximum absolute difference between the NumPy and PyTorch gradients was calculated for:

W1
b1
W2
b2

A very small difference indicates that the manually implemented backpropagation is correct.

# Part 2 - Baseline Model and Activation Study

A PyTorch MLP with at least two hidden layers was trained using four activation functions:

Sigmoid
Tanh
ReLU
Leaky ReLU

All other settings were kept fixed so that the activation functions could be compared fairly.

The validation loss curves for all four models were plotted on a single figure.

The mean absolute gradient of the first hidden layer was recorded for:

Epoch 1
Final epoch

This measurement was performed for both:

Sigmoid
ReLU

For the ReLU model, the percentage of hidden units that output zero for every sample in a validation batch was also calculated.

These results were used to investigate:

Vanishing gradients
ReLU gradient behaviour
Dead ReLU units
# Part 3 - Loss Functions

The classifier was trained using two loss functions:

Categorical cross-entropy
Mean squared error using one-hot encoded targets

Test accuracy was recorded for both models.

The two training curves were plotted together.

A separate tabular regression dataset was also selected and used to train a small MLP.

The regression model was evaluated using:

MSE
RMSE
MAE

The results were used to explain why categorical cross-entropy is generally more suitable for classification than mean squared error.

# Part 4 - Optimiser Comparison

The same network architecture was trained using four optimisers:

SGD
SGD with momentum
RMSProp
Adam

Two experiments were performed.

Same Learning Rate

The same learning rate was used for all four optimisers.

Tuned Learning Rate

A separate learning rate was selected for each optimiser.

The following information was recorded:

Optimiser	Learning Rate	Epochs to Reach 85% Validation Accuracy	Final Validation Accuracy	Wall-clock Time
SGD				
SGD + momentum				
RMSProp				
Adam				

Training loss curves for all four optimisers were plotted on the same figure.

The optimiser was selected based on the experimental results.

# Part 5 - Forcing Overfitting

The purpose of this part was to deliberately create a model with a large generalisation gap.

The training set was reduced to 2,000 samples.

The network was increased to at least four hidden layers with 512 units per layer.

The model was trained until the training accuracy exceeded 99%.

Training and validation loss were plotted on the same figure.

The epoch at which the training and validation curves began to separate was identified.

The generalisation gap was calculated as:

Generalisation Gap =
Training Accuracy - Validation Accuracy

The results were used to determine whether the model showed:

High bias
High variance
Both

The conclusion was based on the actual training and validation results.

# Part 6 - Regularisation Study

The overfitted model from Part 5 was used as the starting point.

Each regularisation method was tested separately while keeping the other settings unchanged.

The following methods were tested:

L2 weight decay
L1 penalty
Dropout
Batch normalisation
Early stopping
Data augmentation
More training data
L2 Weight Decay

At least three different lambda values were tested.

The effect of increasing L2 regularisation on the generalisation gap was measured.

L1 Penalty

The percentage of weights below 1e-3 after training was recorded.

Dropout

At least three dropout rates were tested.

The generalisation gap was measured for each dropout rate.

Batch Normalisation

Batch normalisation was added to the network and its effect on training and validation accuracy was measured.

Early Stopping

Early stopping was tested.

The following were recorded:

Patience value
Epoch at which training stopped
Training accuracy
Validation accuracy
Generalisation gap
Data Augmentation

Training images were augmented using:

Random horizontal flip
Small random rotation
More Training Data

The model was retrained using:

10,000 training samples
20,000 training samples
Regularisation Results

The results were presented in one table:

Method	Setting	Train Accuracy	Validation Accuracy	Gap
Baseline	No regularisation			
L2 weight decay				
L1 penalty				
Dropout				
Batch normalisation				
Early stopping				
Data augmentation				
More training data				

The generalisation gap was also plotted against the strength parameter for:

L2 weight decay
Dropout

This allows the effect of over-regularisation to be observed.

The method that provided the largest reduction in the generalisation gap for the smallest loss in training accuracy was identified.

# Part 7 - Hyperparameter Tuning with k-Fold Cross-Validation

Random search was used to select the final model.

At least three hyperparameters were included in the search space.

Example hyperparameters include:

Learning rate
Hidden layer width
Dropout rate

At least 12 different configurations were evaluated.

Each configuration was scored using 5-fold cross-validation on the training data.

The top five configurations were reported with:

Mean cross-validation score
Standard deviation

The best configuration was selected.

The selected model was then retrained on the full training data together with the best regularisation choices found in Part 6.

The held-out test set was used only for the final evaluation.

Final Evaluation

The final model was evaluated using:

Accuracy
Macro precision
Macro recall
Macro F1-score
Confusion matrix

The final model was compared with the Part 2 baseline.

The improvement was calculated in percentage points:

Improvement =
Final Test Accuracy - Part 2 Baseline Accuracy

If the tuned model did not improve over the baseline, the result was reported honestly and an explanation was provided.

Reproducing the Results

The results can be reproduced using the provided Jupyter notebook.

1. Open the Notebook

Open:

neural_network.ipynb

in Kaggle.

The notebook is designed to run in a Kaggle Notebook environment.

2. Add the Dataset

Add the Fashion-MNIST dataset to the Kaggle notebook using the Add Input option.

Dataset:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

The notebook requires:

fashion-mnist_train.csv
fashion-mnist_test.csv

These files should be available through the Kaggle input directory after the dataset has been added.

3. Select the Accelerator

The intended accelerator is:

GPU T4 × 2

If GPU T4 × 2 is unavailable, another available GPU or CPU can be used.

However, training time may be longer and small differences in numerical results may occur.

4. Required Libraries

The notebook uses:

numpy
pandas
matplotlib
scikit-learn
torch

These libraries are normally available in the Kaggle environment.

5. Set the Random Seed

The notebook uses a fixed random seed:

SEED = 42

The seed should remain unchanged when reproducing the reported results.

6. Run the Notebook in Order

Run the notebook from the first cell to the last cell.

The recommended order is:

Environment setup
        |
        v
Dataset loading
        |
        v
Data preprocessing
        |
        v
Train/validation split
        |
        v
Part 1
        |
        v
Part 2
        |
        v
Part 3
        |
        v
Part 4
        |
        v
Part 5
        |
        v
Part 6
        |
        v
Part 7
        |
        v
Final evaluation

Running cells out of order may result in missing variables or inconsistent results.

7. Data Preprocessing

The notebook performs the following preprocessing:

Original pixel values
0–255
   |
   v
Normalisation
0–1
   |
   v
28 × 28 image
   |
   v
Flattening
   |
   v
784 features

The original test set is not used during model training, validation, or hyperparameter tuning.

8. Training Experiments

Each experiment is run using the settings specified in its corresponding section.

When comparing a specific design choice, other settings are kept fixed where required.

For example, in the activation study, the different activation functions are compared while keeping the rest of the model settings the same.

9. Generated Results

Running the notebook produces:

Dataset sample counts
Class distributions
Training loss curves
Gradient verification results
Activation function comparison
Gradient measurements
ReLU dead-unit percentage
Loss function comparison
Regression metrics
Optimiser comparison table
Overfitting results
Generalisation gap
Regularisation comparison table
L2 generalisation-gap plot
Dropout generalisation-gap plot
Cross-validation results
Final test metrics
Confusion matrix
10. Reproducibility Notes

A fixed random seed is used throughout the notebook.

Small differences may still occur because of:

GPU hardware
PyTorch version
CUDA version
Floating-point calculations
Differences between Kaggle sessions

The results reported in this repository are generated from the provided notebook.

The test set is kept untouched until Part 7 and is used only for the final evaluation.

Results

The actual numerical results are reported in the notebook.

The main results include:

NumPy versus PyTorch gradient differences
Activation function comparison
Vanishing gradient measurements
ReLU dead-unit percentage
Cross-entropy versus MSE
Regression MSE, RMSE and MAE
Optimiser comparison
Overfitting and generalisation gap
L2 regularisation results
L1 regularisation results
Dropout results
Batch normalisation results
Early stopping results
Data augmentation results
More training data results
Hyperparameter search results
Final test accuracy
Macro precision
Macro recall
Macro F1-score
Confusion matrix
# Author

Mehreen Fatima

Assignment: Building, Breaking and Fixing a Neural Network

Year: 2026
