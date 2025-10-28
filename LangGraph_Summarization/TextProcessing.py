import cleantext
from GeminiConfigs import EmbeddingModel
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS


# =========================== READ FILE UTILITY ============================
class ReadFile:
    """
    ReadFile class provides utility methods to read text content from files.
    It supports reading PDF files and plain text files and returning their text content.
    """

    @classmethod
    def read_pdf_files(cls, folder_name: str):
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
            return FAISS.from_documents(chunks, embedding=cls.embeddings, normalize_L2=True)
        except Exception as e:
            print(f"Error in generate_vectors_from_documents: {e}")
            return None

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
        self.file = ReadFile().read_pdf_files(dir_name)

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
if __name__ == "__main__":
    pass