from src.ml_ganAI.components.data_ingestion import DataIngestion
from src.ml_ganAI.components.data_validation import DataValidation
from src.ml_ganAI.components.data_transfomation import DataTransform
from src.ml_ganAI.components.model_trainer import ModelTrainer
from src.ml_ganAI.components.model_evaluation import ModelEvaluation
from src.ml_ganAI.Config.config_entity import DataTransfomationConfig
from src.ml_ganAI import logging



class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        ingestion = DataIngestion()
        ingestion.download_zipFile()
        ingestion.get_zipFile_to_unzip()

class DataValidationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        validation = DataValidation()
        validation.check_columns_validation()    
      
class DataTransformPipeline:
    def __init__(self):
        self.transform = DataTransfomationConfig()
    
    def main(self):
        staus_path = self.transform.status_path
        with open (staus_path,"r") as f:
            status = f.read().split(":")[-1].strip()
            if status != "True":
                logging.error('Data Validation Sataus is False')
            else:    
                transform = DataTransform()
                transform.get_initiate_preprocess()  
                

class ModelTrainPipeline:
    def __init__(self):
        pass
    
    def main(self):
        trainer = ModelTrainer()
        trainer.initiate_model_training()        
        
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        evaluation = ModelEvaluation()
        evaluation.evaluation()                   
                                     
                  
                    
        