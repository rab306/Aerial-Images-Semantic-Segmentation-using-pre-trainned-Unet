Semantic Segmentation Using U-Net
This repository contains the implementation of a semantic segmentation model using the U-Net architecture with a ResNet34 backbone. The project demonstrates the process of training a model to segment images into different classes.
Table of Contents
Introduction
Dataset
Preprocessing
Model Architecture
Training
Results
Installation
Usage
Contributing
License

Introduction
Semantic segmentation is the task of classifying each pixel in an image into a specific class. This project employs a U-Net model with a ResNet34 backbone to segment images into multiple categories.

Dataset
The dataset used for this project consists of images and their corresponding masks. The images are segmented into various classes such as Building, Land, Road, Vegetation, Water, and Unlabeled.
https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery


Preprocessing
The preprocessing steps include:

Rescaling images to the range [0, 1].
Extracting non-overlapping patches of size 256x256 from the images and masks.
Converting RGB masks to categorical labels.
Model Architecture
The model is based on the U-Net architecture with a ResNet34 backbone. Custom loss functions (Dice Loss, Jaccard Loss, and Categorical Cross-Entropy) are used to optimize the model.

Training
The training process involves:

Splitting the dataset into training and validation sets.
Applying data augmentation techniques such as rotation, width shift, height shift, zoom, and flips.
Using callbacks (ReduceLROnPlateau and EarlyStopping) to optimize the training process.
Training the model using the Adam optimizer and custom loss functions.


Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to improve the codebase