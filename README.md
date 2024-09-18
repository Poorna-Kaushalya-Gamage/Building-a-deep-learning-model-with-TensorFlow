
# Building a Deep Learning Model with TensorFlow

## Introduction

This project involved classifying a tabular dataset containing 480 student records, with 17 attributes per record. TensorFlow was used to build a deep learning model for classification. Key steps include data preprocessing, encoding categorical variables, and scaling numerical features. A Sequential deep learning model was developed and trained to improve classification accuracy.

## Methodology

### Data Preparation and Preprocessing
- **Importing Libraries**: Essential libraries were used for data handling, modeling, and assessment.
- **Dataset Handling**: The dataset was imported and cleaned, handling null values, duplicates, garbage values, and typographical errors.
- **Data Visualization**: Statistical summaries and visualization methods were used to understand data distribution.
- **Categorical Encoding and Scaling**: Categorical variables were encoded, and numerical features were scaled using a Min-Max scaler.

### Model Building
- **Splitting the Dataset**: The dataset was split into training and validation sets to evaluate the model.
- **Model Architecture**: A deep learning model was created using TensorFlow’s Sequential API. The architecture included convolutional layers, pooling layers, dense layers, and a final output layer with a Softmax activation function.
  
### Hyperparameter Tuning
Hyperparameters such as learning rate and batch size were tuned to optimize the model's performance.

### Model Evaluation
The following metrics were used to evaluate the model's performance:
- Accuracy
- Precision
- Recall
- F1 Score
- Log Loss

## Findings

Two models were compared based on their performance metrics:

**Model 01**  
- Learning Rate: 0.001   |  Batch Size: 32  
- Accuracy: 0.7604  
- Precision: 0.7612  
- Recall: 0.7604  
- F1 Score: 0.7601  
- Log Loss: 0.6347  

**Model 02**  
- Learning Rate: 0.005  |  Batch Size: 64  
- Accuracy: 0.7083  
- Precision: 0.7231  
- Recall: 0.7083  
- F1 Score: 0.7017  
- Log Loss: 0.6659  

The optimal model, **Model 01**, achieved an accuracy of 0.7604, demonstrating superior performance compared to Model 02.

## Challenges Faced
- **Balancing Metrics**: Difficulty was encountered when trying to balance metrics such as accuracy and log loss.
- **Overfitting**: Overfitting was a concern during model training, but steps were taken to mitigate this issue.
