# Engine Condition Prediction using Hybrid LSTM-CNN Model


This repository contains code for an engine condition prediction project using a hybrid LSTM-CNN model. The model is designed to predict the health condition of engines based on various input features. This documentation provides an overview of the code and its functionalities.






## Purpose
The purpose of this code is to demonstrate the implementation of a hybrid LSTM-CNN model for predicting engine health based on input features. The model is trained and evaluated using a synthetic dataset of engine data.

## Dependencies
Before running the code, make sure you have the following dependencies installed:

pandas: ``` !pip install pandas ```

numpy: ```!pip install numpy```

torch: ```!pip install torch```

scikit-learn: ```!pip install scikit-learn```


## Code Structure
The code is structured into several sections, each serving a specific purpose. Here's an overview of each section:

**Importing Dependencies:** The necessary libraries are imported, including pandas, numpy, torch, and scikit-learn.

**Custom Dataset:** This section defines a custom dataset class, CustomDataset, which is responsible for loading the CSV data and preparing it for training.

**Data Loading and Splitting:** The code loads the dataset from a CSV file, splits it into train, validation, and test sets using train_test_split from scikit-learn, and creates data loaders for each set.

**Hybrid LSTM-CNN Model:** This section defines the architecture of the hybrid LSTM-CNN model, HybridModel, which combines LSTM and CNN layers to make predictions.

**Model Testing:** The Tester class is defined, which evaluates the model's performance on the test set, calculating various metrics such as accuracy, precision, recall, F1 score, AUC, and confusion matrix.

**Cross-Validation Loop:** This section performs a cross-validation loop, training and evaluating the model on different folds of the data to assess its generalization performance.

**Average Evaluation Metrics:** The average evaluation metrics across all folds are calculated and printed, providing an overview of the model's overall performance.

**Final Testing:** The best model from the cross-validation is used to perform a final evaluation on the test set, and the evaluation metrics are printed.

## Usage
To use this code, follow these steps:

Install the necessary dependencies mentioned in the "Dependencies" section.


Replace the csv_file variable with the file path to the data on your machine or in your code. For example: csv_file = 'path/to/your/data.csv'.

Customize the hyperparameters and model architecture if needed, such as input_size, hidden_size, num_classes, and num_epochs.

Optionally, modify the early stopping criteria, batch size, and other training settings to suit your needs.

Run the code to train and evaluate the model. The average evaluation metrics across all folds will be printed, followed by the evaluation metrics on the final test set using the best model.








## Screenshots
Here are some screenshots of the code in action:

Training and Evaluation Logs: This screenshot showcases the training and evaluation process, displaying the training and evaluation loss for each epoch.

![image](https://github.com/DancesWithDobes/engine_failure_prediction/assets/69741804/7efd5d02-222b-40cb-812b-905afde6bb34)



Average Validation Metrics: This screenshot shows the average evaluation metrics across all folds, including loss, accuracy, precision, recall, F1 score, AUC, and the confusion matrix.


![image](https://github.com/DancesWithDobes/engine_failure_prediction/assets/69741804/0441868f-e871-4608-9de2-1d1990231e0f)



Final Test Metrics: This screenshot presents the evaluation metrics on the final test set using the best model, including loss, accuracy, precision, recall, F1 score, AUC, and the confusion matrix.


![image](https://github.com/DancesWithDobes/engine_failure_prediction/assets/69741804/ef976af6-786d-43eb-b9ac-67e12d0f0010)




## Conclusion
This Colab file provides a comprehensive implementation of a hybrid LSTM-CNN model for engine condition prediction. 


(Note to potential employeers / interviewers.. it may be easier to use the link in my resume to run the file directly in my colab)






