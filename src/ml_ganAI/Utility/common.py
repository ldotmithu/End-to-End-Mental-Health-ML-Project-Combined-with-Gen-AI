from pathlib import Path
from src.ml_ganAI import logging
import os 
import yaml

def Create_Folder(file_path):
    try:
        os.makedirs(file_path,exist_ok=True)
        logging.info(f"{file_path} Created")
    except  Exception as e:
        raise e     
    
def Read_Yaml(File_path):
    try:
        with open(File_path,'r') as f:
            file= yaml.safe_load(f)
            logging.info(f"{File_path} Read the yaml")
            return file
            
    except Exception as e:
        raise e 
        
def check_xls_occur(dir_path):
    files = os.listdir(dir_path)
    xls_file = [file for file in files if file.endswith(".xls")] 
    if len(xls_file) == 1:
            return xls_file[0]
    elif len(xls_file) == 0:
        logging.error("Don't have any xls files")
        return None
    else:
        logging.error("Multipule xls files are there")
        return None  