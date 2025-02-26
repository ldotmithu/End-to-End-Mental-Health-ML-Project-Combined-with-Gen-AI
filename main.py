from src.ml_ganAI.Pipeline.Stages_of_Pipeline import DataIngestionPipeline,DataValidationPipeline
from src.ml_ganAI import logging

try:
    logging.info(">>>>>>Data Ingestion>>>>>>>")
    ingestion = DataIngestionPipeline()
    ingestion.main()
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
except Exception as e:
    raise e 

try:
    logging.info(">>>>>>Data Validation>>>>>>>")
    validation = DataValidationPipeline()
    validation.main()
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
except Exception as e:
    raise e 

