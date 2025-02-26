import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from src.ml_ganAI.Config.config_entity import DataTransfomationConfig
from src.ml_ganAI import logging
from src.ml_ganAI.Utility.common import Create_Folder, Read_Yaml, check_xls_occur, remove_out
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
            ("num_pipeline", StandardScaler())
            ])
        
        power_pipeline = Pipeline([
            ('power_pipeline', PowerTransformer(method="yeo-johnson")
             
             )])
        
    
        preprocessor = ColumnTransformer([
            ("power", power_pipeline, power_columns),
            ("num", num_pipeline, num_columns)
        ])
        return preprocessor
    
    def get_initiate_preprocess(self):
        
        csv_file = check_xls_occur(self.transform.data_path)
        if not csv_file:
            logging.error("No valid .xls file found in the specified directory.")
            return None
        
        data = pd.read_csv(os.path.join(self.transform.data_path, csv_file))
        
        # Remove outliers 
        data = remove_out(data, 'Age')
        data = remove_out(data, 'SystolicBP')
        data = remove_out(data, 'BS')
        data = remove_out(data, 'HeartRate')
        
      
        data['RiskLevel'] = np.where(data['RiskLevel'] == "high risk", 2,
                             np.where(data['RiskLevel'] == "mid risk", 1, 0))
        
        # Split into train and test datasets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
      
        target_col = self.schema.get("TARGET")
        
   
        train_input_feature = train_data.drop(columns=target_col, axis=1)
        train_target_feature = train_data[target_col]
        
        test_input_feature = test_data.drop(columns=target_col, axis=1)
        test_target_feature = test_data[target_col]
        
        # Initialize the preprocessing 
        preprocess_obj = self.initiate_preprocess()
        train_pre = preprocess_obj.fit_transform(train_input_feature)
        test_pre = preprocess_obj.transform(test_input_feature)
        
        # Apply SMOTEENN 
        smt = SMOTEENN(random_state=42, sampling_strategy='auto')
        train_pre, train_target_feature = smt.fit_resample(train_pre, train_target_feature)
        
       
        train_arr = np.c_[train_pre, np.array(train_target_feature)]
        test_arr = np.c_[test_pre, np.array(test_target_feature)]
        
        # Save processed data
        np.save(os.path.join(self.transform.root_dir, "train.npy"), train_arr)
        np.save(os.path.join(self.transform.root_dir, "test.npy"), test_arr)
        
        logging.info("Data preprocessing completed and files saved.")
        
        joblib.dump(preprocess_obj,os.path.join(self.transform.root_dir,self.transform.preprocess_path))
        logging.info("Data preprocessing files saved in pkl .")
        
        
