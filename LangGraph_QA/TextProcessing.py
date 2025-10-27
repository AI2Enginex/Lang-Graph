from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from LLMConfigs import GeminiConfig, EmbeddingModel
from langchain_community.vectorstores import FAISS
import cleantext
# ========================== FILE UTILITIES ============================

class ReadFile:
    """
    Utility to read text from PDFs or plain text files.
    """
    @classmethod
    def read_pdf_file(cls, folder_name=None):
        try:
            text = ""
            with open(folder_name, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

# ========================== TEXT CHUNKING ============================

class TextChunks:
    """
    Handles splitting text into chunks.
    """
    text_splitter = None

    @classmethod
    def initialize(cls, separator=None, chunksize=None, overlap=None):
        try:
            cls.text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=chunksize,
                chunk_overlap=overlap
            )
        except Exception as e:
            print(f"Failed to initialize splitter: {e}")

    @classmethod
    def get_text_chunks_doc(cls, text=None):
        try:
            if cls.text_splitter is None:
                print("Text splitter not initialized!")
                return None
            return cls.text_splitter.create_documents([text])
        except Exception as e:
            print(f"Error creating document chunks: {e}")
            return None

# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config: GeminiConfig, api_key: str):
        try:
            cls.embeddings = EmbeddingModel(config=config, api_key=api_key).embeddings
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")

    @classmethod
    def generate_vectors_from_documents(cls, chunks=None):
        try:
            if cls.embeddings is None:
                print("Embedding model not initialized!")
                return None
            return FAISS.from_documents(chunks, embedding=cls.embeddings)
        except Exception as e:
            print(f"Error generating vectors: {e}")
            return None

# ========================== PREPARE TEXT ============================


class PrepareText:
    """
    Prepares and processes text from PDF files.
    Handles reading, cleaning, chunking, and vectorization.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str):
        try:
            self.file_path = file_path
            self.config = config
            self.api_key = api_key
            self.raw_text = ReadFile.read_pdf_file(file_path)
        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.raw_text = ""

    def clean_data(self):
        try:
            return cleantext.clean(
                self.raw_text,
                lowercase=True,
                punct=True,
                extra_spaces=True
            )
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return self.raw_text

    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        try:
            TextChunks.initialize(separator=separator, chunksize=chunksize, overlap=overlap)
            return TextChunks.get_text_chunks_doc(text=self.clean_data())
        except Exception as e:
            print(f"Error creating chunks: {e}")
            return []

    def create_text_vectors(self, separator=None, chunksize=None, overlap=None):
        try:
            Vectors.initialize(config=self.config, api_key=self.api_key)
            return Vectors.generate_vectors_from_documents(
                chunks=self.get_chunks(separator, chunksize, overlap)
            )
        except Exception as e:
            print(f"Error creating vectors: {e}")
            return None
if __name__ == "__main__":
    pass