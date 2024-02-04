# Vegetable Image Classification and Quantity Estimation - DataThon (Problem statement 5 - DataByte)

This project implements an image classification model to identify vegetables from images. It also integrates object detection to estimate purchase quantities and nutrition information.

## Overview

The goals of this project are:

- Accurately classify images of vegetables into 15 categories
- Detect and count individual instances of vegetables
- Estimate total purchase quantities needed for meal prep
- Calculate nutrition information like weights for planning
- Provide an interactive GUI for easy use

The system combines computer vision techniques like convolutional neural networks and object detection with a GUI for usability.

## Data

The model is trained on a dataset of over 5000 vegetable images categorized into 15 classes:

- Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato

The data is split 70/15/15 into training, validation and test sets. The training data is augmented with shifts, flips, rotations etc. to reduce overfitting.

## Model Architecture

The classifier uses a MobileNetV2 model pretrained on ImageNet. The base layers are frozen and a global average pooling layer and dense layers are added on top as the classifier head. This is trained to classify the 15 vegetable categories.

For object detection, a Roboflow Faster R-CNN model trained on a separate dataset is used.

## Usage

The GUI allows easy interactive usage with ipywidgets. Users can enter a list of vegetables and quantities to track. Then on submitting an image path, the classifier will identify the vegetables present. The object detector counts the instances and updates the remaining quantities needed. The total weight is also calculated using average weights from nutritional data.

## Installation

The required packages are:

```
tensorflow==2.5.0
keras==2.6.0
pandas==1.1.5 
ipywidgets==7.6.5
etc.
```

The trained classification model file is also included as `final_model.h5`.

## Results

The classification model achieves 99.7% accuracy on the test set and 99.4 on the validation set.

## Future Work

Some ways to improve the system further:

- Increase classification accuracy with more training data, hyperparameter tuning etc. 
- Support detection and counting of more vegetable categories.
- Improve nutrition estimates using a larger nutrition dataset.
- Deploy the GUI as a web application.
- Optimize the object detector for faster inference.

Please feel free to work on any of these ideas by forking and improving the project!

## References

The dataset is from https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

The object counting/validation model was created with [Roboflow](https://roboflow.com)
