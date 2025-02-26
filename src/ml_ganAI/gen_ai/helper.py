from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain,LLMChain,RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

prompt_template = PromptTemplate(
    input_variables=["input_data", "prediction"],
    template="""
    You are a Mental Health Risk Medical Assistant. Your role is to explain why a specific risk level (High, Middle, or Low) was predicted based on the user's input and provide simple, actionable suggestions to address it. Use easy-to-understand language and focus on the most relevant factors contributing to the risk level.

    ### **Patient Data & Risk Level Prediction:**
    - Input Data: {input_data}
    - Predicted Risk Level: {prediction}

    ### **Steps to Follow:**
    1. **Explain the Risk Level**: Briefly describe why the user's input led to the predicted risk level. Highlight the most significant factors (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate) that contributed to the prediction.
    2. **Provide Suggestions**: Offer clear, practical, and personalized recommendations based on the risk level. Include food suggestions, entertainment ideas, if the user is at High or Middle risk. If the user is at Low risk, provide a positive greeting and general wellness tips.

    ### **Example Format:**
    - **Risk Level**: [High/Middle/Low]
    - **Why This Happened**: [Brief explanation of the risk factors.]
    - **Suggestions**: [Actionable steps, food suggestions]

    ### **Response:**
    """
)
llm = ChatGroq(temperature=0.5, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

rag_chain = LLMChain(llm=llm, prompt=prompt_template)

# Input data and prediction
input_data = [[29,90,70,8,100,80]]  # Example input
prediction = model.predict(input_data)  # Get prediction from your ML model

# Convert input and prediction to strings
input_data_str = str(input_data)
prediction_str = str(prediction[0])

# Generate response using the RAG chain
response = rag_chain.invoke({"input_data": input_data_str, "prediction": prediction_str})
print(response["text"])