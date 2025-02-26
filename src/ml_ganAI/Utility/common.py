from pathlib import Path
from src.ml_ganAI import logging
import os 
import yaml
import numpy as np 

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
    
def remove_out(data,col):
    Q1 = np.percentile(data[col],25)
    Q3 = np.percentile(data[col],75)
    IQR = Q3-Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 +1.5 *IQR
    data = data[(data[col] > lower) & (data[col] < upper)]
    logging.info(f"remove the outlier {col}")
    return data    