from src.ml_ganAI.Config.config_entity import DataValidationConfig
import os
from src.ml_ganAI import logging
from src.ml_ganAI.constant.constant_config import *
from src.ml_ganAI.Utility.common import Create_Folder,Read_Yaml,check_xls_occur
import pandas as pd 


class DataValidation:
    def __init__(self):
        self.validation = DataValidationConfig()
        self.schema = Read_Yaml(SCHEMA_PATH)['COLUMNS']
        
        Create_Folder(self.validation.root_dir)
    
    def check_columns_validation(self):
        try:
            csv_path = check_xls_occur(self.validation.data_path)
            data = pd.read_csv(os.path.join(self.validation.data_path,csv_path))
            all_col = list(data.columns)
            req_col = self.schema.keys()
            
            miss_col = [col for col in req_col if col not in all_col]
            if miss_col:
                logging.error(f"missing columns are :{miss_col}")
                validation_status = False
                with open(self.validation.status_path,"w") as f:
                    f.write(f"Validation_status : {validation_status}")
                    logging.info(f"Validation_status : {validation_status}")
            else:
                validation_status = True
                with open(self.validation.status_path,"w") as f:
                    f.write(f"Validation_status : {validation_status}")
                    logging.info(f"Validation_status : {validation_status}")
                    
        except Exception as e:
            raise e            
                    
            