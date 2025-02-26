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
    status_path:str = "artifacts/data_validation/Status.txt"
    
    
