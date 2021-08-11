# galaxy-classification

## Introduction

## Data
The data are publicly available <a href="https://astronn.readthedocs.io/en/latest/galaxy10.html">here</a>.  The data file is too large to store on GitHub. To use the code in this repo, download the .h5 file from the link provided and store it in the data folder.

The dataset contains 17,300 256 x 256 images of galaxies, categorized into 10 classes.  The class labels are as follows:

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

Although the maximum and minimum class totals differ by an order of magnitude, the data are still fairly balanced.

## Data Preparation

The following steps were taken to prepare the data for ingest by the neural network:

1. Data were divided into a 90/10 train-test-split (stratified split used to account for variation in class totals)
2. Images were downsampled from 256x256 to 128x128 to reduce computational burden
3. Centers of images were cropped to omit unnecessary background pixels

## Models

### Baseline
A simple baseline model was created for reference using Scikit-learn's DummyClassifer.  This model always predicts the most frequent class in the training set.

### Convolutional Neural Network

### Transfer Learning

## Results
The results of the baseline model and the highest performing CNN are shown in the table below.

<center>
  
| Model | Accuracy |
| ----- | ---------|
| Baseline| 14.9%|
| CNN | 81.1 %|

</center>

### CNN Analysis

The confusion matrix below provides insight into the CNN's performance on the test set.

<img src="https://github.com/jstodd867/galaxy-classification/blob/main/imgs/confusion_matrix.png">

The values on the diagonal represent the percentage of the true class occurrences that were correctly predicted.  From these numbers, it is clear that the classifier performs worst on classes 0, 4, 6, and 7.  These classes correspond to:  disturbed, cigar-shaped smooth, unbarred tight spiral, and unbarred loose spiral.  Taking a closer look at these classes, it is not surprising that they have the highest misclassification rates.  The latter two categories are very similar and it is difficult for the human eye to distinguish these from each other.  The cigar-shaped smooth category is very similar in appearance to the third category (in-between round smooth); distinguishing between these two categories is also tough for the human eye.  The confusion matrix shows that the largest percentage of misclassifications, 15%, are attributed to the in-between round smooth class.  Finally, the disturbed class by far has the highest error rates.  Not only is this the class with the smallest relative amount of instances inthe data set, but it is also visually difficult to distinguish from several of the other classes.  A sample of 10 images from this class are shown below, for comparison with the plot in the Data Exploration section.

<img src="https://github.com/jstodd867/galaxy-classification/blob/main/imgs/Disturbed_sample.png">

## Conclusions and Next Steps

The convolutional neural network designed and trained for classifying this data performed with 81% accuracy.  As would be expected, transfer learning utilizing popular pre-trained convolutional neural networks like VGG16 and Inception did not perform as well.

Next steps include:

<li> Targeted Data Augmentation - Increasing the artificial augmentation of the training data with scaled and rotated versions of the classes with the highest misclassification rates (e.g., disturbed, cigar-shaped smooth)</li>
<li> Ensemble methods incorporating other model types </li>
<li> Completing a Flask app that predicts the galaxy type for an uploaded image </li>
