# Import necessary modules
import os
import sys
import pandas as pd
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.utils.utils import load_object

# Define the PredictPipeline class
class PredictPipeline:
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_data = preprocessor.transform(features)
            predictions = model.predict(scaled_data)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise customexception(e, sys)

# Define the CustomData class
class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.error('Exception occurred while creating dataframe.')
            raise customexception(e, sys)

# Main function for generating predictions
def generate_predictions():
    try:
        # Create an instance of CustomData with input data
        data = CustomData(carat=0.5, depth=60.0, table=58.0, x=5.0, y=5.0, z=3.0, cut='Ideal', color='E', clarity='SI1')
        
        # Convert input data to DataFrame
        input_df = data.get_data_as_dataframe()
        
        # Instantiate PredictPipeline and generate predictions
        predictor = PredictPipeline()
        predictions = predictor.predict(input_df)
        
        # Print predictions
        print("Predictions:", predictions)
        
    except customexception as ce:
        logging.error(f"Custom Exception: {str(ce)}")
    except Exception as e:
        logging.error(f"Unknown Error: {str(e)}")

# Call the main function to generate predictions
if __name__ == "__main__":
    generate_predictions()
