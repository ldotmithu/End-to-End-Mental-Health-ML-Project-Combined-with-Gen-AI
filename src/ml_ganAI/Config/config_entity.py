import os 
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir:Path = "artifacts/data_ingestion"
    URL:str = "https://github.com/ldotmithu/Dataset/raw/refs/heads/main/Maternal%20Health%20Risk%20Data%20Set.zip"
    local_data_path:Path =  "artifacts/data_ingestion/data.zip"
    unzip_dir:Path = "artifacts/data_ingestion"
    
@dataclass
class DataValidationConfig:
    root_dir:Path = "artifacts/data_validation"
    data_path:Path = "artifacts/data_ingestion"
    status_path:Path = "artifacts/data_validation/Status.txt"

@dataclass
class DataTransfomationConfig:
    root_dir:Path = "artifacts/data_transform"
    status_path:Path = "artifacts/data_validation/Status.txt"
    data_path:Path = "artifacts/data_ingestion"
    preprocess_path:str = "preprocess.pkl"
    
@dataclass
class ModelTrainerConfig:
    root_dir:Path= "artifacts/trainer" 
    train_data_path:Path = "artifacts/data_transform/train.npy"
    model_path = "model.pkl"
      
        
    
    
