from pathlib import Path
import joblib

model_path = Path("artifacts/model_trainer/model.pkl")
preprocess_path = Path("artifacts/data_transform/preprocess.pkl")

class Predication_Pipeline:
    def __init__(self):
        # Load the model and preprocessing pipeline
        self.model = joblib.load(model_path)
        self.preprocessing_pipeline = joblib.load(preprocess_path)
        
    def transform(self, data):
        """Preprocess the data using the loaded preprocessing pipeline"""
        try:
            transformed_data = self.preprocessing_pipeline.transform(data)
            return transformed_data
        except Exception as e:
            raise ValueError(f"Error transforming data: {str(e)}")
    
    def prediction(self, data):
        """Predict the outcome using the loaded model"""
        try:
            prediction = self.model.predict(data)
            return prediction
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")