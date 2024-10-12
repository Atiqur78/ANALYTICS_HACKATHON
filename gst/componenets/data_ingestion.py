import os
from gst.exception import CustomException
from gst.logger import logging
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', "train.csv")
        self.test_data_path = os.path.join('artifacts', "test.csv")
        self.raw_data_path = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method start")
        try:
            df_train = pd.read_csv(os.path.join(r'D:\GST_HACKATHON\notebooks\data', 'X_Train_Data_Input.csv'))
            df_train_target = pd.read_csv(os.path.join(r'D:\GST_HACKATHON\notebooks\data', 'Y_Train_Data_Target.csv'))
            df_train["target"] = df_train_target["target"]
            logging.info('Dataset read as pandas Dataframe')

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df_train.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Raw data is created')

            # Split the data into training and test sets
            train_set, test_set = train_test_split(df_train, test_size=0.30, random_state=42)

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"First Five row of train dataset: \n {train_set.head()}")
            logging.info(f"First Five row of test dataset: \n {test_set.head()}")

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
