import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split
from src.ml_ganAI.Config.config_entity import DataTransfomationConfig
from src.ml_ganAI import logging
from src.ml_ganAI.Utility.common import Create_Folder, Read_Yaml, check_xls_occur
from src.ml_ganAI.constant.constant_config import *
import joblib

class DataTransform:
    def __init__(self):
        self.transform = DataTransfomationConfig()
        self.schema = Read_Yaml(SCHEMA_PATH)
        Create_Folder(self.transform.root_dir)
    
    def initiate_preprocess(self):
        num_columns = self.schema.get("NUM_COLUMNS", [])
        power_columns = self.schema.get("POWER", [])
        
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])
        
        power_pipeline = Pipeline([
            ('power', PowerTransformer(method="yeo-johnson"))
        ])
        
        preprocessor = ColumnTransformer([
            ("power", power_pipeline, power_columns),
            ("num", num_pipeline, num_columns)
        ])
        
        return preprocessor
    
    def handle_outliers(self, data, column):
       
        lower_bound = data[column].quantile(0.05)
        upper_bound = data[column].quantile(0.95)
        data[column] = np.clip(data[column], lower_bound, upper_bound)
        return data
    
    def get_initiate_preprocess(self):
       
        csv_file = check_xls_occur(self.transform.data_path)
        if not csv_file:
            logging.error("No valid .xls file found in the specified directory.")
            return None
        
        data = pd.read_csv(os.path.join(self.transform.data_path, csv_file))

     
        outlier_columns = ['Age', 'SystolicBP', 'BS', 'HeartRate']
        for col in outlier_columns:
            data = self.handle_outliers(data, col)
        
        
        data['RiskLevel'] = np.where(data['RiskLevel'] == "high risk", 2,
                                     np.where(data['RiskLevel'] == "mid risk", 1, 0))
        
        target_col = self.schema.get("TARGET")

       
        X = data.drop(columns=[target_col])
        y = data[target_col]

       
        preprocess_obj = self.initiate_preprocess()
        X_preprocessed = preprocess_obj.fit_transform(X)

        
        smt = SMOTE(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = smt.fit_resample(X_preprocessed, y)

        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

       
        np.save(os.path.join(self.transform.root_dir, "train.npy"), np.c_[X_train, np.array(y_train)])
        np.save(os.path.join(self.transform.root_dir, "test.npy"), np.c_[X_test, np.array(y_test)])
        
        logging.info("Data preprocessing completed and files saved.")
        
       
        joblib.dump(preprocess_obj, os.path.join(self.transform.root_dir, self.transform.preprocess_path))
        logging.info("Preprocessing pipeline saved as a pickle file.")
