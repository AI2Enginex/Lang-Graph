from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai  # Importing the Google Generative AI module from the google package
import os
import cleantext
# import time
# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = "xxxxxxxxxxxxxxxxxxxxxx"

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')


# defining a class QAState
# inherits BaseModel

class QAState(BaseModel):
    question: str
    retrieved_chunks: List[str]
    answer: str
    prompt_type: str  
                        

class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
        self.llm=ChatGoogleGenerativeAI(temperature=0.7,model="gemini-1.5-flash", google_api_key=key,top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=3000)


class EmbeddingModel:
    def __init__(self, model_name):
        # Initializing GoogleGenerativeAIEmbeddings object with the specified model name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)

class GenerateContext(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)

    def generate_response(self, query):
        try:
            # Generating response content based on the query using the inherited model
            return [response for response in self.model.generate_content(query)]
        except Exception as e:
            return e


# =========================== READ FILE UTILITY ============================
class ReadFile:
    """
    ReadFile class provides utility methods to read text content from files.
    It supports reading PDF files and plain text files and returning their text content.
    """

    @classmethod
    def read_file_text(cls, folder_name=None):
        """
        Reads and extracts text from a PDF file.

        Args:
            folder_name (str): The path to the PDF file to be read.

        Returns:
            str: The combined text extracted from all pages of the PDF file.
            Exception: Returns the exception object if an error occurs while reading the file.
        """
        try:
            text = ""  # Initialize an empty string to store extracted text
            
            # Open the PDF file in binary read mode
            with open(folder_name, 'rb') as file:
                reader = PdfReader(file)  # Create a PdfReader object to parse the PDF file
                
                # Iterate through each page in the PDF and extract text
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()  # Append text from each page
                    
            return text  # Return the extracted text from the entire PDF
        except Exception as e:
            # In case of any exception (e.g., file not found, read error), return the exception
            return e

    @classmethod
    def read_file_and_store_elements(cls, filename):
        """
        Reads a plain text file line by line, strips whitespace, and concatenates the content.

        Args:
            filename (str): The path to the text file to be read.

        Returns:
            str: A single string containing all lines concatenated without leading/trailing spaces.
            Exception: Returns the exception object if an error occurs while reading the file.
        """
        try:
            text = ''  # Initialize an empty string to store the cleaned file content
            
            # Open the text file in read mode
            with open(filename, "r") as file:
                
                # Read each line in the file
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace and newline characters
                    text += line  # Append the cleaned line to the text variable
                    
            return text  # Return the concatenated text content
        except Exception as e:
            # In case of any exception (e.g., file not found), return the exception
            return e


# =========================== TEXT CHUNKING UTILITY ============================

class TextChunks:
    """
    Handles splitting of text data into smaller, manageable chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = None  # Class variable to hold the text splitter instance

    @classmethod
    def initialize(cls, separator=None, chunksize=None, overlap=None):
        """
        Initializes the text splitter with specified separator, chunk size, and overlap.
        
        Args:
            separator (list): List of separators used to split text.
            chunksize (int): Maximum size of each chunk.
            overlap (int): Overlap size between consecutive chunks.
        """
        try:
            # Initialize RecursiveCharacterTextSplitter with provided parameters
            cls.text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=chunksize,
                chunk_overlap=overlap
            )
            print("Text splitter initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize text splitter: {e}")
            cls.text_splitter = None  # Reset splitter on failure

    @classmethod
    def get_text_chunks(cls, text=None):
        """
        Splits the given text into smaller chunks using the initialized text splitter.
        
        Args:
            text (str): Input text to split.
        
        Returns:
            list: List of text chunks.
        """
        if cls.text_splitter is None:
            print("Text splitter is not initialized! Call initialize() first.")
            return None
        try:
            return cls.text_splitter.split_text(text)
        except Exception as e:
            print(f"Error splitting text: {e}")
            return None

    @classmethod
    def get_text_chunks_doc(cls, text=None):
        """
        Splits the given text into document chunks (structured for LLM processing).
        
        Args:
            text (str): Input text to split.
        
        Returns:
            list: List of document chunks (as LangChain Documents).
        """
        if cls.text_splitter is None:
            print("Text splitter is not initialized! Call initialize() first.")
            return None
        try:
            return cls.text_splitter.create_documents([text])
        except Exception as e:
            print(f"Error creating document chunks: {e}")
            return None

# =========================== VECTOR STORE UTILITY ============================

class Vectors:
    """
    Handles generation of vector embeddings from text or document chunks using a specified embedding model.
    """
    embeddings = None  # Class variable to hold the embedding model instance

    @classmethod
    def initialize(cls, model_name):
        """
        Initializes the embedding model.
        
        Args:
            model_name (str): Name or type of the embedding model.
        """
        try:
            cls.embeddings = EmbeddingModel(model_name=model_name)
            print(f"Embedding model initialized with {model_name}")
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            cls.embeddings = None  # Reset embeddings on failure

    @classmethod
    def generate_vectors_from_text(cls, chunks=None):
        """
        Generates vector embeddings from text chunks and stores them in FAISS.
        
        Args:
            chunks (list): List of text chunks.
        
        Returns:
            FAISS: FAISS vector store containing embeddings.
        """
        if cls.embeddings is None:
            print("Embedding model is not initialized!")
            return None
        try:
            return FAISS.from_texts(chunks, embedding=cls.embeddings.embeddings)
        except Exception as e:
            print(f"Error in generate_vectors_from_text: {e}")
            return None

    @classmethod
    def generate_vectors_from_documents(cls, chunks=None):
        """
        Generates vector embeddings from document chunks and stores them in FAISS.
        
        Args:
            chunks (list): List of document chunks.
        
        Returns:
            FAISS: FAISS vector store containing embeddings.
        """
        if cls.embeddings is None:
            print("Embedding model is not initialized!")
            return None
        try:
            return FAISS.from_documents(chunks, embedding=cls.embeddings.embeddings)
        except Exception as e:
            print(f"Error in generate_vectors_from_documents: {e}")
            return None

# =========================== PROMPT TEMPLATES ============================

## The given Prompt-Templates are focused for Question and Anaswering

class PromptTemplates:
    """
    This class provides reusable prompt templates for different prompting strategies:
    1. Keyword Extraction (RAG style factual QA)
    2. Chain-of-Thought reasoning (step-by-step logical answering)
    3. Verification prompts (double-check factual correctness)
    """

    @classmethod
    def key_word_extraction(cls):
        """
        Use Case:
        ---------
        Ideal for RAG (Retrieval-Augmented Generation) tasks.
        It ensures that the model answers strictly based on retrieved document context
        without adding external knowledge or assumptions.

        Example Usage:
        - Document Question Answering.
        - Information extraction tasks where accuracy is critical.
        - Preventing hallucinations in factual QA.

        Returns:
            A PromptTemplate instructing the model to:
            - ONLY use facts explicitly from context.
            - Avoid external knowledge/assumptions.
            - Reply explicitly if the answer is not present.
        """
        try:
            prompt = """
                    You are an intelligent assistant.

                    Below is the information retrieved from the document:

                    {context}

                    Now, answer the following question strictly based on the above information:
                    Try to elaborate the answer if possible.
                    
                    Question: {question}

                    Guidelines:
                    - ONLY use facts explicitly mentioned in the context.
                    - Do NOT use external knowledge.
                    - Do NOT make assumptions.
                    - If the answer is not present, reply: "The document does not contain this information."

                    Provide your answer:
                    """
            return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])
        except Exception as e:
            return e

    @classmethod
    def chain_of_thoughts(cls):
        """
        Use Case:
        ---------
        Implements Chain-of-Thought (CoT) prompting.
        Helps model break down complex questions and reason step by step logically based ONLY on the document content.

        Example Usage:
        - Complex document QA needing logical reasoning.
        - Scenarios where multi-step analysis improves answer accuracy.
        - Summarization tasks requiring explanation or derivation.

        Returns:
            A PromptTemplate guiding the model to:
            - Think step by step.
            - Extract, analyze, and logically derive the answer.
            - Avoid external knowledge and assumptions.
            - Avoid bullet points; present a smooth, clear final answer.
        """
        try:
            prompt = """
                       You are a thoughtful assistant.

                        Here is the document content:

                        {context}

                        Question: {question}

                        Think step by step based ONLY on the provided content.
                        Extract the relevant information, analyze it logically, and derive a clear answer.

                        Rules:
                        - Do NOT use any outside knowledge.
                        - Do NOT assume facts not explicitly stated.
                        - Just only display and elaborate the final answer.

                        Begin reasoning:
                    """
            return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])
        except Exception as e:
            return e

    @classmethod
    def verification_prompt(cls):
        """
        Use Case:
        ---------
        Implements Verification prompting technique.
        Helps double-check the factual correctness of model-generated answers by verifying if they are supported by the document content.

        Example Usage:
        - Post-processing stage after answer generation to ensure factual alignment.
        - Use in sensitive domains like healthcare, legal, finance.
        - To flag unverifiable or hallucinated answers.

        Returns:
            A PromptTemplate that:
            - Cross-verifies if the provided answer is directly supported by document context.
            - Gives a binary "Verified" or "Cannot verify" response.
        """
        try:
            prompt = """
                        You are a careful assistant.

                        Here is the document content:

                        {context}

                        Question: {question}

                        Provide the answer based ONLY on the above document content.

                        Verify if the answer can be directly supported by the content.
                        - If YES, state: "Verified: Answer supported by the document."
                        - If NO, state: "Cannot verify: The document does not contain enough information."

                        Answer:
                    """
            return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])
        except Exception as e:
            return e
        
class PromptManager:
    def __init__(self):
        """
        Initializes a dictionary that maps prompt names to their respective methods
        in the PromptTemplates class.
        """
        self.prompt_dict = {
            "key word extraction": PromptTemplates.key_word_extraction,
            "chain of thoughts": PromptTemplates.chain_of_thoughts,
            "verification prompt": PromptTemplates.verification_prompt
        }

    def get_prompt(self, prompt_name):
        """
        Retrieves the appropriate prompt template based on user input.

        Args:
            prompt_name (str): The name of the prompt template to retrieve.

        Returns:
            PromptTemplate instance if the prompt exists, otherwise raises ValueError.
        """
        prompt_function = self.prompt_dict.get(prompt_name)
        if not prompt_function:
            raise ValueError(f"Prompt '{prompt_name}' not found! Available prompts: {list(self.prompt_dict.keys())}")
        return prompt_function()  # Call the function to get the PromptTemplate

class PrepareText:
    """
    Prepares and processes text from files.
    Handles reading, cleaning, chunking, and vectorization of text.
    """

    def __init__(self, dir_name):
        """
        Constructor to read text from a file (PDF).

        Args:
            dir_name (str): Path to the directory/file containing the document.
        """
        # Reading the raw text from PDF file using ReadFile class
        self.file = ReadFile().read_file_text(dir_name)

    def clean_data(self):
        """
        Cleans the raw text by converting to lowercase, removing punctuation and extra spaces.

        Returns:
            str: Cleaned text.
        """
        try:
            return cleantext.clean(
                self.file,
                lowercase=True,
                punct=True,
                extra_spaces=True
            )
        except Exception as e:
            return e

    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        """
        Splits cleaned text into document chunks.

        Args:
            separator (list): Separators to split text.
            chunksize (int): Max size of each chunk.
            overlap (int): Overlap between chunks.

        Returns:
            list: List of document chunks.
        """
        try:
            # Initialize TextChunks and split cleaned text into document chunks
            TextChunks.initialize(separator=separator, chunksize=chunksize, overlap=overlap)
            return TextChunks.get_text_chunks_doc(text=self.clean_data())
        except Exception as e:
            return e

    def create_text_vectors(self, separator=None, chunksize=None, overlap=None, model=None):
        """
        Generates vector embeddings from the document chunks.

        Args:
            separator (list): Separators to split text.
            chunksize (int): Chunk size.
            overlap (int): Overlap size.
            model (str): Name of embedding model.

        Returns:
            FAISS: Vector store containing document embeddings.
        """
        try:
            # Initialize embedding model and create vectors from document chunks
            Vectors.initialize(model_name=model)
            return Vectors().generate_vectors_from_documents(
                chunks=self.get_chunks(separator, chunksize, overlap)
            )
        except Exception as e:
            return e

# Defining a QAsystem class for Question and Answering 
# the class purpose is to retrieve the chunks and
# answer based on the selected Prompt
class QASystem(PrepareText, ChatGoogleGENAI):

    def __init__(self, filename=None, delimiter=None, chunk=None, over_lap=None, model=None):
        # initialize PrepareText with filename
        PrepareText.__init__(self, dir_name=filename)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self)

        # calling the preprocessed vectors
        self.vector_store = self.create_text_vectors(
            separator=delimiter,           
            chunksize=chunk, 
            overlap=over_lap, 
            model=model
        )
    

    # functions for extracting the chunks
    def retrieve_chunks(self, state: QAState):
        try:
            docs = self.vector_store.similarity_search(state.question, k=4)
            retrieved_chunks = [doc.page_content for doc in docs]

            return QAState(
                question=state.question,
                retrieved_chunks=retrieved_chunks,
                answer=state.answer,
                prompt_type=state.prompt_type
            )
        except Exception as e:
            print("Error in retrieve_chunks:", e)
            return state
    
    def answer_questions(self, state: QAState):
        """
        Uses the selected prompt template to answer a question based on retrieved document chunks.
        """
        try:
            prompt_manager = PromptManager()
            prompt_template = prompt_manager.get_prompt(state.prompt_type)

            context = "\n\n".join(state.retrieved_chunks)
            prompt = prompt_template.format(context=context, question=state.question)
            response = self.llm.invoke(prompt)

            return QAState(
                question=state.question,
                retrieved_chunks=state.retrieved_chunks,
                answer=response.content,
                prompt_type=state.prompt_type
            )

        except Exception as e:
            print("Error in answer_questions:", e)
            return state

# Class for QASystem Execuetion
# creating a Graph Execuetion flow
class QASystemGraphExecuetion(QASystem):

    def __init__(self, data=None, processing_delimiter=None, total_chunk=None, overlapping=None, embedding_model=None):
        """
        Initializes the QASystemGraphExecuetion class.

        Parameters:
        - data (str): Path to the PDF file.
        - processing_delimiter (str): Delimiter to split text.
        - total_chunk (int): Size of each text chunk.
        - overlapping (int): Overlap between chunks.
        - embedding_model (str): Name of embedding model to use.
        """
        # Initialize the parent QASystemGraphExecuetion class with provided parameters
        super().__init__(filename=data, delimiter=processing_delimiter, chunk=total_chunk, over_lap=overlapping, model=embedding_model)
    
    def build_graph(self):
        """
        Builds a LangGraph execution graph for QASystem flow.
        
        Graph structure:
        - Node 1: Retrieve document chunks relevant to the query.
        - Node 2: Fetch the Question and provide the answers.
        - Edge: Connects retrieve → QA.
        """
        try:
            # Create a LangGraph with initial state defined by QAState
            graph = StateGraph(QAState)
            # Add the 'retrieve' node, which fetches document chunks
            graph.add_node("retrieve", self.retrieve_chunks)
            
            # Add the 'QA' node, which fetches the question and returns the answers
            graph.add_node("QA", self.answer_questions)
            
            # Define the execution flow: retrieve → QA
            graph.add_edge("retrieve", "QA")
            
            graph.set_entry_point("retrieve")
            return graph
        except Exception as e:
            #Return the exception if any error occurs
            return e
        
    def answer(self, question: str, prompt: str):
        try:

            # Build the execution graph
            graph_executor = self.build_graph()

            # Compile the graph into an executable object
            executor = graph_executor.compile()
            initial_state = {"question": question, "retrieved_chunks": [], "answer": "", "prompt_type": prompt}
            result = executor.invoke(initial_state)
            return result['answer']
        except Exception as e:
            return e

if __name__ == "__main__":
    
    
    question = input("Ask your Question here: ")
    user_prompt_type = input("Choose prompt type (key word extraction / chain of thoughts / verification prompt): ")
    qa_system = QASystemGraphExecuetion(
                                  data='E:\Lang-Graph\wings_of_fire.pdf',
                                  processing_delimiter='\n\n',
                                  total_chunk=1000,
                                  overlapping=300,embedding_model='models/embedding-001')
    answer = qa_system.answer(question=question, prompt=user_prompt_type)
    
    print("Question:", question)
    print("Answer:", answer)