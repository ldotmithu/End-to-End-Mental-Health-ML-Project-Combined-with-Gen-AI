from src.ml_ganAI.Config.config_entity import DataIngestionConfig
from src.ml_ganAI import logging
from src.ml_ganAI.Utility.common import Create_Folder
import zipfile,os
from urllib import request

class DataIngestion:
    def __init__(self):
        self.ingestion = DataIngestionConfig()
        
        Create_Folder(self.ingestion.root_dir)
        
    def download_zipFile(self):
        exteantion_path = self.ingestion.URL
        if exteantion_path.endswith(".zip"):
            if not os.path.exists(self.ingestion.local_data_path):
                request.urlretrieve(self.ingestion.URL,self.ingestion.local_data_path)
                logging.info("Zip File Downloaded")
            else:
                logging.info("File already exists")
        else:
            logging.error("Invalid URL Extentaion")        
            
    def get_zipFile_to_unzip(self):
        try:
            unzip_path = self.ingestion.unzip_dir
            with zipfile.ZipFile(self.ingestion.local_data_path) as f:
                f.extractall(unzip_path)
                logging.info("Unzip Operation Done")
        except Exception as e:
            raise e             
            
        