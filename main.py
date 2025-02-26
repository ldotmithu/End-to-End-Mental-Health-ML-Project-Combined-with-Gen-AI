from src.ml_ganAI.Pipeline.Stages_of_Pipeline import DataIngestionPipeline
from src.ml_ganAI import logging

try:
    logging.info(">>>>>>Data Ingestion>>>>>>>")
    ingestion = DataIngestionPipeline()
    ingestion.main()
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
except Exception as e:
    raise e 
