# Toxic Comment Classification with CNN

This project is a machine learning solution that classifies toxic comments using a Convolutional Neural Network (CNN). It uses the Jigsaw Toxic Comment Classification dataset and pre-processes the text data before passing it through the model. The aim is to detect whether a comment contains any form of toxicity (toxic, severe toxic, obscene, threat, insult, identity hate).

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Results](#results)

## Project Overview
This project leverages a deep learning approach for text classification to predict whether comments in a dataset are toxic or non-toxic. Using a CNN-based model, we preprocess raw text data, clean it, and tokenize it before feeding it into the network for classification. The model is evaluated using metrics such as accuracy, ROC-AUC score, and a classification report.

## Technologies Used
- **Programming Languages**: Python
- **Deep Learning**: TensorFlow, Keras
- **Natural Language Processing (NLP)**: NLTK
- **Machine Learning**: Scikit-learn
- **Data Manipulation**: Pandas, NumPy

## Dataset
The dataset used for this project is the [Jigsaw Toxic Comment Classification Challenge dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). It includes the following toxic categories for labeling:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

### Data Files:
- **train.csv**: Contains the training data with text and labels.
- **test.csv**: Contains the test data with text.

### Data Preprocessing:
- Text data is cleaned by removing non-alphabetic characters and stopwords.
- Tokenization is performed, and sequences are padded to ensure uniform length.
  
## Model Architecture
The model used for this project is a **Convolutional Neural Network (CNN)** with the following architecture:
- **Embedding Layer**: Converts the text data into dense vectors.
- **Conv1D Layer**: Applies 1D convolution with a kernel size of 5.
- **GlobalMaxPooling1D Layer**: Reduces the dimensionality of the features.
- **Dense Layers**: Two fully connected layers, one with 64 units and ReLU activation, and the output layer with sigmoid activation for binary classification.
- **Dropout**: Added to reduce overfitting.

### Model Summary:
```plaintext
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 500, 128)          2560000   
 conv1d (Conv1D)             (None, 496, 128)          82048     
 global_max_pooling1d (GlobalMaxPooling1D)  (None, 128)   0         
 dense (Dense)               (None, 64)                8256      
 dropout (Dropout)           (None, 64)                0         
 dense_1 (Dense)             (None, 1)                 65        
=================================================================
Total params: 2,645,369
Trainable params: 2,645,369
Non-trainable params: 0
```
## Project Structure
```
├── cleaned_train_data.csv        # Preprocessed training data
├── cleaned_test_data.csv         # Preprocessed test data
├── test_predictions.csv          # Test set predictions after running the model
├── toxic_comment_classification.py   # Main Python script
├── train.csv                     # Training dataset (toxic comments)
├── test.csv                      # Test dataset (toxic comments)
├── README.md                     # Project readme file
└── toxicity_cnn_model.keras      # Saved Keras model
```
## Results
After training the model, it outputs:

- **Accuracy**: Accuracy for training and validation sets.
 - **Classification Report**: Precision, Recall, and F1-score for the validation set.
- **ROC-AUC Score**: Measure of model performance.
Sample output for POC-AUC score:
```
Validation ROC-AUC Score: 0.94
```
Here’s a classification report for the validation set:
```
              precision    recall  f1-score   support

           0       0.97      0.98      0.98      4784
           1       0.80      0.72      0.76       441

    accuracy                           0.96      5225
   macro avg       0.88      0.85      0.87      5225
weighted avg       0.96      0.96      0.96      5225
```


