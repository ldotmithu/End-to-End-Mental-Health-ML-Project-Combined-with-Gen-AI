from src.ml_ganAI.Config.config_entity import ModelEvaluationConfig
from src.ml_ganAI import logging
import numpy as np 
from src.ml_ganAI.Utility.common import Create_Folder,save_object
import joblib,os
from sklearn.metrics import accuracy_score


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation = ModelEvaluationConfig()
        
        Create_Folder(self.model_evaluation.root_dir)
        
    def evaluation(self):
        test_data = np.load(self.model_evaluation.test_data_path)
        
        test_input_feature = test_data[:,:-1]
        test_target_feature = test_data[:,-1]    
        
        model_obj = joblib.load(self.model_evaluation.model_path)
        prd = model_obj.predict(test_input_feature)
        acc= accuracy_score(test_target_feature,prd)
        save_object(file_path=os.path.join(self.model_evaluation.root_dir,self.model_evaluation.metrics_path),
                    obj=acc)
        logging.info(f"acc_score :{acc} ")
        