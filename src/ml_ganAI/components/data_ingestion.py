from src.ml_ganAI.Config.config_entity import DataIngestionConfig
from src.ml_ganAI import logging


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        