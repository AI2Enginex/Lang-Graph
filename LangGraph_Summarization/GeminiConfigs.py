from typing import Annotated

from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings



import google.generativeai as genai  # Importing the Google Generative AI module from the google package
import os


# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key =  "AIzaSyC4m4mJdC4Meic3W6501FjdFX-giYFgE28"
os.environ['GOOGLE_API_KEY'] = api_key
# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')


class SimpleDocState(TypedDict):
    messages: Annotated[list, "add_messages"]  
    document_chunks: list  
                        
class ReducedDocState(TypedDict):
    messages: Annotated[list, "add_messages"]  
    document_chunks: list                    
    partial_summaries: list

class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = genai.GenerationConfig(
            temperature=0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.llm=ChatGoogleGenerativeAI(temperature=0,
                                        model="gemini-2.5-flash", 
                                        google_api_key=key,
                                        top_p=0.9,
                                        top_k=32,
                                        max_output_tokens=3000)


class EmbeddingModel:
    def __init__(self, model_name):
        # Initializing GoogleGenerativeAIEmbeddings object with the specified model name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)

class GenerateContext(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)

    def generate_response(self, query):
        try:
            # Generating response content based on the query using the inherited model
            return [response for response in self.model.generate_content(query)]
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    pass