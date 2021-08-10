# galaxy-classification

## Introduction

## Data
The data are publicly available <a href="https://astronn.readthedocs.io/en/latest/galaxy10.html">here</a>.  The dataset contains 17,300 256 x 256 images of galaxies, categorized into 10 classes.  The class labels are as follows:

0. Disturbed
1. Merging
2. Round Smooth
3. In-between Round Smooth
4. Cigar Shaped Smooth
5. Barred Spiral
6. Unbarred Tight Spiral
7. Unbarred Loose Spiral
8. Edge-on Without Bulge
9. Edge-on With Bulge

## Exploratory Data Analysis

An example image from each class is shown in the image below.

<img src="https://github.com/jstodd867/galaxy-classification/blob/main/imgs/class_examples.png">

Before building any models, I plotted the distribution of the class occurrences in the data to determine whether any measures would need to be taken to deal with imbalanced data.  This plot is shown below.

<img src="https://github.com/jstodd867/galaxy-classification/blob/main/imgs/class_occurrences.png">

## Data Cleaning and Preparation

## Models

### Baseline

### Convolutional Neural Network

## Results
The results of the baseline model and the highest performing CNN are shown in the table below.

| Model | Accuracy |
| ----- | ---------|
| Baseline| 14.9%|
| CNN | 81.1 %|

### CNN Analysis

The confusion matrix below provides insight into the CNN's performance on the test set.

<img src="https://github.com/jstodd867/galaxy-classification/blob/main/imgs/confusion_matrix.png">

The values on the diagonal represent the percentage of the true class occurrences that were correctly predicted.  From these numbers, it is clear that the classifier performs worst on classes 0, 4, 6, and 7.  These classes correspond to:  disturbed, cigar-shaped smooth, unbarred tight spiral, and unbarred loose spiral.

