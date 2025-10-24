from typing import Annotated
from PyPDF2 import PdfReader
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai  # Importing the Google Generative AI module from the google package
import os
import cleantext
import time
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
        self.llm=ChatGoogleGenerativeAI(temperature=0.7,model="gemini-2.5-flash", google_api_key=key,top_p=1.0,
            top_k=32,
            candidate_count=1,
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
            raise e

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
            raise e


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
            cls.embeddings = EmbeddingModel(model_name=model_name).embeddings
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
            return FAISS.from_texts(chunks, embedding=cls.embeddings)
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
            return FAISS.from_documents(chunks, embedding=cls.embeddings)
        except Exception as e:
            print(f"Error in generate_vectors_from_documents: {e}")
            return None

# =========================== PROMPT TEMPLATES ============================

class PromptTemplates:
    """
    Contains different prompt templates for summarization tasks.
    """
    @classmethod
    def summarisation_chains(cls):
        """Direct summarization prompt for smaller documents."""
        try:
            prompt_template = """
            You are given a PDF document.
            Your job is to generate a concise summary of the given document in 15 points.
            Try to cover each point from starting till the end.
            Try to summarize each line in the document.

            Context:
            {context}

            Answer:
            """
            return PromptTemplate(template=prompt_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e
        
    @classmethod
    def summarisation_prompt(cls):
        """
        Direct summarization prompt template for summarizing small documents with Strict Constraints.
        """
        try:
            prompt_template = """
            You are given a PDF document .
            Your job is to generate a concise summary of the given document in not more than 200 words.
            Try to give a brief summary by explaining each and every point from the satarting till the end.
            Display the summary as if you are giving a presentation.

            Try to summarize each line in the document.

            Context:
            {context}

            Answer:
            """
            return PromptTemplate(template=prompt_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

    @classmethod
    def map_prompt(cls):
        """
        Map prompt template: Summarizes each chunk of a document individually.
        """
        try:
            map_template = """
            You are given a part (chunk) of a PDF document.
            Summarize the key points of this chunk, try to summarize each point in brief.
            Avoid adding information not present in the chunk.

            Document Chunk:
            {context}

            Chunk Summary:
            """
            return PromptTemplate(template=map_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

    @classmethod
    def reduce_prompt(cls):
        """
        Reduce prompt template: Combines individual chunk summaries into a cohesive final summary.
        """
        try:
            reduce_template = """
            You are provided with multiple chunk-level summaries of a PDF document.
            Combine these summaries into a clear, cohesive final summary.
            Do not repeat points; focus on merging and refining.

            Partial Summaries:
            {context}

            Final Summary:
            """
            return PromptTemplate(template=reduce_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

# =========================== TEXT PREPARATION CLASS ============================

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
            raise e

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
            raise e

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
            raise e

class StuffSummarizer(PrepareText,ChatGoogleGENAI):
    """
    StuffSummarizer is a class designed to perform direct summarization 
    (also known as "stuff" summarization) on PDF documents.

    This class uses LangChain's LLMChain and PromptTemplate to generate a concise summary 
    of a document in a single pass, without breaking it down into chunks or applying 
    MapReduce-style summarization.

    It inherits:
    - ChatGoogleGENAI: For LLM capabilities.
    - PrepareText: For text reading, cleaning, chunking, and vector generation.
    """

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
    
    # function to retrieve the chunks
    def retrieve_chunks(self, state: SimpleDocState):
        try:
            query = state["messages"][-1]
            results = self.vector_store.similarity_search(query, k=3)
            return {
                "messages": state["messages"],
                "document_chunks": results,  # Ensure chunks are strings
            }
        except Exception as e:
            raise e
        
    # summarization function
    def summarize_chunks(self,state: SimpleDocState):
        """Summarize extracted chunks using LLM."""
        try:
            content = "\n\n".join([chunk.page_content for chunk in state["document_chunks"]])
            prompt_template = PromptTemplates.summarisation_prompt()
            prompt = prompt_template.format(context=content)
            response = self.llm.invoke(prompt)   # returns a BaseMessage
            summary = response.content # Extracting the message
            return {"messages": state["messages"] + [summary], "document_chunks": state["document_chunks"]}
        except Exception as e:
            raise e

class MapReduceSummarizer(PrepareText,ChatGoogleGENAI):
    """
    MapReduce is a class designed to perform MapReduce-style summarization
    on PDF documents.

    It inherits:
    - ChatGoogleGENAI: Provides access to LLM capabilities.
    - PrepareText: Handles reading, cleaning, chunking, and vector creation from PDF text.

    This class applies the MapReduce summarization technique by:
    1. Breaking the document into chunks.
    2. Summarizing each chunk (map step).
    3. Combining partial summaries into a cohesive final summary (reduce step).
    """

    def __init__(self, filename=None, delimiter=None, chunk=None, over_lap=None, model=None):
        # initialize PrepareText with filename
        PrepareText.__init__(self, dir_name=filename)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self)

        # Create vector store by chunking and embedding the document
        self.vector_store = self.create_text_vectors(
            separator=delimiter,
            chunksize=chunk,
            overlap=over_lap,
            model=model
        )
    
    # function to retrieve chunks
    def retrieve_chunks(self, state: ReducedDocState):
        try:
            query = state["messages"][-1]  # Last message is query string
            results = self.vector_store.similarity_search(query, k=5)

            return {
                "messages": state["messages"],
                "document_chunks": [chunk.page_content for chunk in results],  # List of strings
                "partial_summaries": []
            }
        except Exception as e:
            raise e
    
    # Map stage function
    def map_summarize(self, state: ReducedDocState):
        try:

            print('Starting Map Node')
            partial_summaries = []
            prompt_template = PromptTemplates.map_prompt()

            for chunk in state["document_chunks"]:
                formatted_prompt = prompt_template.format(context=chunk)
                summary = self.llm.invoke(formatted_prompt)  # returns a BaseMessage

                # Extract plain strig from the BaseMessage
                partial_summaries.append(summary.content)

            return {
                "messages": state["messages"],
                "document_chunks": state["document_chunks"],
                "partial_summaries": partial_summaries
            }
        except Exception as e:
            raise e
    
    # Reduce stage function
    def reduce_summarize(self, state: ReducedDocState):
        try:

            print('Starting Reduce node')
            # Join all partial summaries into one combined string
            combined = "\n\n".join(state["partial_summaries"])
            prompt_template = PromptTemplates.reduce_prompt()
            formatted_prompt = prompt_template.format(context=combined)

            final_summary = self.llm.invoke(formatted_prompt) # returns a BaseMessage

            return {
                "messages": state["messages"] + [final_summary.content],  # Append only the string content
                "document_chunks": state["document_chunks"],
                "partial_summaries": state["partial_summaries"]
            }
        except Exception as e:
            raise e


# Class for StuffSummarization
# creating a Graph Execuetion flow
class StuffGraphExecuetion(StuffSummarizer):

    def __init__(self, data=None, processing_delimiter=None, total_chunk=None, overlapping=None, embedding_model=None):
        """
        Initializes the StuffGraphExecuetion class.

        Parameters:
        - data (str): Path to the PDF file.
        - processing_delimiter (str): Delimiter to split text.
        - total_chunk (int): Size of each text chunk.
        - overlapping (int): Overlap between chunks.
        - embedding_model (str): Name of embedding model to use.
        """
        # Initialize the parent StuffSummarizer class with provided parameters
        super().__init__(filename=data, delimiter=processing_delimiter, chunk=total_chunk, over_lap=overlapping, model=embedding_model)

    def build_graph(self):
        """
        Builds a LangGraph execution graph for direct summarization.
        
        Graph structure:
        - Node 1: Retrieve document chunks relevant to the query.
        - Node 2: Summarize the retrieved chunks.
        - Edge: Connects retrieve → summarize
        """
        try:
            # Create a LangGraph with initial state defined by SimpleDocState
            graph = StateGraph(SimpleDocState)

            # Add the 'retrieve' node, which fetches document chunks
            graph.add_node("retrieve", self.retrieve_chunks)

            # Add the 'summarize' node, which performs summarization
            graph.add_node("summarize", self.summarize_chunks)

            # Define the execution flow: retrieve → summarize
            graph.add_edge("retrieve", "summarize")

            # Set 'retrieve' node as the entry point of the graph
            graph.set_entry_point("retrieve")

            return graph
        except Exception as e:
            # Return the exception if any error occurs
            raise e

    def summarize(self, query: str):
        """
        Executes the LangGraph to summarize the document based on the user query.

        Parameters:
        - query (str): The query or instruction for summarization.

        Returns:
        - Final summarized text (str).
        """
        try:
            # Build the execution graph
            graph_executor = self.build_graph()

            # Compile the graph into an executable object
            executor = graph_executor.compile()

            # Initial state: contains the user query and empty document chunks
            initial_state = {
                "messages": [query],
                "document_chunks": [],
            }

            # Run the graph executor with the initial state
            final_state = executor.invoke(initial_state)

            # Return the last message, which should be the final summary
            return final_state["messages"][-1]
        except Exception as e:
            # Return the exception if an error occurs
            raise e


# Class for MapReduceSummarization
# creating a Graph Execuetion flow
class MapReduceGraphExecuetion(MapReduceSummarizer):

    def __init__(self, data=None, processing_delimiter=None, total_chunk=None, overlapping=None, embedding_model=None):
        """
        Initializes the MapReduceGraphExecuetion class.

        Parameters:
        - data (str): Path to the PDF file.
        - processing_delimiter (str): Delimiter to split text.
        - total_chunk (int): Size of each text chunk.
        - overlapping (int): Overlap between chunks.
        - embedding_model (str): Name of embedding model to use.
        """
        # Initialize the parent MapReduceSummarizer class with provided parameters
        super().__init__(filename=data, delimiter=processing_delimiter, chunk=total_chunk, over_lap=overlapping, model=embedding_model)
    
    def build_graph(self):
        """
        Builds a LangGraph execution graph for MapReduce-style summarization.
        
        Graph structure:
        - Node 1: Retrieve document chunks relevant to the query.
        - Node 2: Apply Map summarization on each chunk.
        - Node 3: Apply Reduce summarization on the partial summaries to generate final output.
        """
        try:
            print('Now Executing MapReduce Graph')

            # Create a LangGraph with initial state defined by ReducedDocState
            graph = StateGraph(ReducedDocState)

            # Add the 'retrieve' node to fetch document chunks
            graph.add_node("retrieve", self.retrieve_chunks)

            # Add the 'map_summarize' node to summarize each chunk individually
            graph.add_node("map_summarize", self.map_summarize)

            # Add the 'reduce_summarize' node to combine partial summaries into final summary
            graph.add_node("reduce_summarize", self.reduce_summarize)

            # Define execution flow: retrieve → map_summarize → reduce_summarize
            graph.add_edge("retrieve", "map_summarize")
            graph.add_edge("map_summarize", "reduce_summarize")

            # Set the entry point to the 'retrieve' node
            graph.set_entry_point("retrieve")

            return graph
        except Exception as e:
            # raise exception if any error occurs
            raise e

    def summarize(self, query: str):
        """
        Executes the LangGraph to summarize the document using MapReduce technique.

        Parameters:
        - query (str): The query or instruction for summarization.

        Returns:
        - Final summarized text (str).
        """
        try:
            # Build the execution graph
            graph_executor = self.build_graph()

            # Compile the graph into an executable object
            executor = graph_executor.compile()

            # Define the initial state:
            # - messages: user query
            # - document_chunks: empty, to be filled during retrieval
            # - partial_summaries: to store intermediate chunk-level summaries
            initial_state = {
                "messages": [query],
                "document_chunks": [],
                "partial_summaries": []
            }

            # Run the graph executor with the initial state
            final_state = executor.invoke(initial_state)

            # Return the final summarized message
            return final_state["messages"][-1]
        except Exception as e:
            # raise exception if error occurs
            raise e
        
if __name__ == "__main__":
    
    chain_type = input("Enter chain type: ")
    query = "Summarize this document briefly."
    
    if chain_type == 'simple':
        rag = StuffGraphExecuetion(data='E:/Lang-Graph/RILAGM.pdf',processing_delimiter='\n\n',total_chunk=5000,overlapping=300,embedding_model='models/gemini-embedding-001')
        summary = rag.summarize(query=query)
        print("Summary:\n", summary)
    else:
        rag = MapReduceGraphExecuetion(data='E:/Lang-Graph/RILAGM.pdf',processing_delimiter='\n\n',total_chunk=5000,overlapping=100,embedding_model='models/gemini-embedding-001')
        summary = rag.summarize(query=query)
        print("Summary:\n", summary)
