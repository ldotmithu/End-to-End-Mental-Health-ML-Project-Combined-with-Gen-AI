from src.ml_ganAI.components.data_ingestion import DataIngestion
from src.ml_ganAI.components.data_validation import DataValidation


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
      
        