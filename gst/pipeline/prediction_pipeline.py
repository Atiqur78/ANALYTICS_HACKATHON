import os
import pickle as pkl
import numpy as np
import pandas as pd
from gst.exception import CustomException
from gst.logger import logging
import sys
from gst import COLUMN_TO_DROP

def prediction(file_name):
    try:
        model = pkl.load(open('artifacts/model.pkl', 'rb'))
        scaler = pkl.load(open('artifacts/preprocessor.pkl', 'rb'))
        data = pd.read_csv(file_name)
        data = data.drop(COLUMN_TO_DROP, axis=1)
        data = scaler.transform(data)
        prediction = model.predict(data)
        pred_prob = 
        return prediction
    except Exception as e:
        logging.info('Error Occured in prediction')
        raise CustomException(e, sys)
