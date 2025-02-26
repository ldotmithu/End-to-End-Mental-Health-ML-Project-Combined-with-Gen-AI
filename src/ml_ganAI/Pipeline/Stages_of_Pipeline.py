from src.ml_ganAI.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        ingestion = DataIngestion()
        ingestion.download_zipFile()
        ingestion.get_zipFile_to_unzip()