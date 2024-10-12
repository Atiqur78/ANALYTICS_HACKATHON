import os
import sys
from gst.logger import logging
from gst.exception import CustomException
import pandas as pd

from gst.componenets.data_ingestion import DataIngestion
from gst.componenets.data_transformation import DataTransformation
from gst.componenets.model_trainer import ModelTrainer

if __name__ =='__main__':
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(train_data_path, test_data_path)

        data_transformation = DataTransformation()
        train_arr, test_arr,_ =data_transformation.initiate_data_transforamtion(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
