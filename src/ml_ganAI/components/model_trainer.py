from src.ml_ganAI.Config.config_entity import ModelTrainerConfig
from src.ml_ganAI import logging
from xgboost import XGBClassifier
import numpy as np 
from src.ml_ganAI.Utility.common import Create_Folder,Read_Yaml
from src.ml_ganAI.constant.constant_config import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib,os


class ModelTrainer:
    def __init__(self):
        self.trainer = ModelTrainerConfig()
        self.perams = Read_Yaml(PARAMS_PATH)['Model']
            
        Create_Folder(self.trainer.root_dir)
    
    def initiate_model_training(self):
        train_data = np.load(self.trainer.train_data_path)
        
        train_data_input_feature = train_data[:,:-1]
        train_data_target_feature = train_data[:,-1] 
          
    
        #knn = KNeighborsClassifier()
        xgb = XGBClassifier(min_child_weight=self.perams.get("min_child_weight"),
                            max_depth = self.perams.get("max_depth"))
        
        xgb.fit(train_data_input_feature,train_data_target_feature)
        #print(xgb.score(train_data_input_feature,train_data_target_feature))
        
        joblib.dump(xgb,os.path.join(self.trainer.root_dir,self.trainer.model_path))
        logging.info("save the model.pkl")       