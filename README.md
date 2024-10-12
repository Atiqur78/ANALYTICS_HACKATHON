# GST Analytics Hackathon
==========================

`Note: I have uploaded the project as a ZIP file, the checksum has been generated of that file `

Model Evaluation and Its Impact
Introduction
This document outlines the process, methodology, and performance evaluation of the machine learning models developed for the binary classification task. The models were trained and tested using the provided dataset, and their performances were assessed on various metrics such as accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, and log loss. The following sections detail the approach, methodology, and findings.
Model Code and Documentation
The model training and testing code has been written using Python and the scikit-learn library. Various classification algorithms such as Random Forest, Decision Tree, XGBoost, AdaBoost, and CatBoost were evaluated. Additionally, data preprocessing steps like handling missing values, scaling, and oversampling using SMOTE were applied. Detailed methodology is as follows:
Methodology
Data Preprocessing:
1. Missing values were handled using mean imputation.
2. Correlation analysis was performed to drop highly correlated features (threshold > 0.8).
3. StandardScaler was used to standardize the dataset.
4. SMOTE was applied to handle data imbalance, which resampled the dataset to ensure balanced class distribution.
Models Evaluated
The following models were evaluated during the process:
            
**Model Name   	f1_score**
1. Random Forest	0.985974
2. CatBoosting Classifier	0.984806
3. XGBClassifier	0.984230
4. Gradient Boosting   	0.982924
5. AdaBoost Classifier   	0.982772
6. Decision Tree  	0.979064


**Model Performance Report**
The models were evaluated based on multiple performance metrics to assess their classification abilities. The metrics evaluated include:
1. Accuracy: The proportion of correctly classified instances (both true positives and true negatives) out of the total instances.
2. Precision: The proportion of true positive instances out of the instances predicted as positive.
3. Recall: The proportion of true positive instances out of the actual positive instances.
4. F1 Score: The harmonic mean of precision and recall.
5. AUC-ROC: Measures the model’s ability to distinguish between classes.
6. Log Loss: Measures the performance of a classification model where the predicted output is a probability value.
7. Confusion Matrix: Provides a breakdown of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

**Final Model Selection**

Based on the evaluation of the above metrics, the Random Forest model was selected for final predictions. It performed well across various metrics, balancing both recall and precision, and had a good AUC-ROC score and low log loss.
Results
The final Random Forest model was evaluated on the test dataset, and the following results were obtained:
1. Accuracy: 0.9735663630250045
2. Precision: 0.7897037713689156
3. Recall: 0.9808736526460815
4. F1 Score: 0.8749683715886499
5. AUC-ROC Score: 0.9768396208588458
6. Log Loss: 0.0727324898661461
 

