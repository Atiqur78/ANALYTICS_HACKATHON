import pickle as pkl
import os, sys
import pandas  as pd
import numpy as np
from gst.logger import logging
from gst.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb')as file_obj:
            pkl.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(models,X_train, y_train, X_test, y_test):
    try:
        report={}
        for model_name, model in models.items():

            #train model
            model.fit(X_train, y_train)

            #prediction Testing data
            y_test_pred = model.predict(X_test)

            y_test_prob = model.predict_proba(X_test)

            # Get Accuracy  scores for the train and test data
            accuracy =accuracy_score(y_test, y_test_pred)
            F1_Score = f1_score(y_test, y_test_pred)
            Precison_Score = precision_score(y_test, y_test_pred)
            Recall_Score = recall_score(y_test, y_test_pred)
            Log_Loss = log_loss(y_test, y_test_prob)

            

            report[model_name] =  {
                'accuracy': accuracy,
                'f1_score': F1_Score,
                'precision': Precison_Score,
                'recall': Recall_Score,
                'log_loss': Log_Loss
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)
