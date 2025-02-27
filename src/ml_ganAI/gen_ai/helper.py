from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import os
from dotenv import load_dotenv
from src.ml_ganAI.Pipeline.prediction import Predication_Pipeline  
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


prediction_pipeline = Predication_Pipeline()

class GenAI:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.5, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
        
    def system_prompt(self):


        prompt_template = PromptTemplate(
            input_variables=["input_data", "prediction"],
            template="""
            You are a Mental Health Risk Medical Assistant. Your role is to **analyze the user's health data** and explain why a specific risk level was predicted. Then, provide **structured, actionable suggestions** to improve their mental and physical well-being.

            ---
            ### 🏥 Patient Data & Risk Level Prediction:
            - **Input Data:** {input_data}
            - **Predicted Risk Level:** {prediction} (0 = Low, 1 = Medium, 2 = High)

            ---
            ### 📌 **Risk Assessment Summary**
            Risk Level :- [High Level / Medium Level / Low Level]

            🔍 Why This Happened:
            - [Explain why this risk level was assigned]

            ---
            ### ✅ **Personalized Health Recommendations**
            #### 🔹 **Actionable Steps**  
            1. [Step 1: Practical lifestyle or medical advice]
            
            2. [Step 2: A simple daily improvement tip]

            #### 🍏 Food Recommendations 
            Healthy food suggestions for mental and physical well-being  

            #### 🎭 Entertainment & Wellness Ideas  
            [Suggestions like meditation, light exercise, or stress-relieving activities  
            
            ### ### 😊 if risk level is low provide greeting 
            [If the prediction is low level, I'll just send a greeting or a wish to maintain the same level. If advice is needed, I'll provide it. Let me know what you need!😊
            
            ---
            ### 📝 Example Response:
            
            🏥 Risk Level :- **High Level**  

            🔍 Why This Happened:
            -  

            ---
            ### ✅ Personalized Health Recommendations
            #### 🔹 **Actionable Steps**  
            1. 
            2. 

            #### 🍏 Food Recommendations  
            -
            -

            #### 🎭 Entertainment & Wellness Ideas
            -
            -

            """
            
        )


        return prompt_template
    
    def response(self, input_data):
        try:
            column_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
            prompt_template = self.system_prompt()

            input_data_df = pd.DataFrame(input_data, columns=column_names)

       
            input_data_processed = prediction_pipeline.transform(input_data_df)

          
            predicted_value = prediction_pipeline.prediction(input_data_processed)[0]

            
            rag_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = rag_chain.invoke({"input_data": input_data, "prediction": predicted_value})
            return response["text"]
        except Exception as e:
            return f"Error generating response: {str(e)}"