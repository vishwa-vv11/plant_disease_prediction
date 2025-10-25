Project Overview

This project aims to detect and classify plant diseases from leaf images using Convolutional Neural Networks (CNNs). Early detection of plant diseases can help farmers take timely measures to prevent crop loss and increase yield. The system allows users to upload an image of a leaf and instantly receive information about the disease affecting the plant.
______________________________________________________________________________________________________________________________________________________________________________
Key Features:

Classifies plant diseases using leaf images.

Supports multiple plant types and disease classes.

Provides a simple GUI for uploading leaf images.

Shows the predicted disease name and probability.

Easy to deploy and use via a web interface (Streamlit).
______________________________________________________________________________________________________________________________________________________________________________
Dataset

The model is trained using the PlantVillage dataset, which contains images of healthy and diseased plant leaves for multiple crops.

Dataset Processing:

Images are resized to a uniform size.

Normalization applied to pixel values.

Data is split into training, validation, and testing sets.
______________________________________________________________________________________________________________________________________________________________________________
Tools & Libraries

->Python 3.10+

->TensorFlow / Keras – for building the CNN model

->OpenCV – for image preprocessing

->NumPy / Pandas – for data handling

->Matplotlib / Seaborn – for visualizing results

->Streamlit – for creating the GUI and deploying the model
______________________________________________________________________________________________________________________________________________________________________________
Model Architecture

The model is a CNN trained to recognize multiple plant disease classes:

Input Layer: 128x128x3 images

Convolutional Layers + MaxPooling

Fully Connected (Dense) Layers

Output Layer: Softmax activation for multi-class classification
______________________________________________________________________________________________________________________________________________________________________________
Performance Metrics:

Accuracy on training and testing sets

Confusion matrix visualization

Loss curves during training
