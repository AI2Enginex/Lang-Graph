from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from LLMConfigs import GeminiConfig, EmbeddingModel
import cleantext


# ========================== FILE UTILITIES ============================

class ReadFile:
    """
    Utility to read text from PDFs or plain text files.
    """

    @classmethod
    def read_pdf_file(cls, file_path: str) -> str:
        """
        Reads and extracts text content from a PDF file.
        Cleans basic layout noise (line breaks, hyphenation).

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Combined text from all pages.
        """
        try:
            text_parts = []
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = (
                            page_text.replace("-\n", "")
                            .replace("\n", " ")
                            .replace("\t", " ")
                        )
                        page_text = " ".join(page_text.split())
                        text_parts.append(page_text)
            return " ".join(text_parts)
        except Exception as e:
            print(f"Error reading PDF file '{file_path}': {e}")
            return ""


# ========================== TEXT CHUNKING ============================

class TextChunks:
    """
    Handles splitting text into smaller chunks for LLM/embedding processing.
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
            print(f"Text splitter initialized (chunk={chunksize}, overlap={overlap}, separators={separator})")
        except Exception as e:
            print(f"Failed to initialize text splitter: {e}")
            cls.text_splitter = None

    @classmethod
    def get_text_chunks_doc(cls, text=None):
        try:
            if cls.text_splitter is None:
                print("Text splitter not initialized. Call initialize() first.")
                return []
            return cls.text_splitter.create_documents([text])
        except Exception as e:
            print(f"Error creating document chunks: {e}")
            return []


# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config: GeminiConfig):
        """
        Initializes the embedding model from the Gemini configuration.
        """
        try:
            cls.embeddings = EmbeddingModel(config=config).embeddings
            if cls.embeddings:
                print(f"Embedding model loaded: {config.embedding_model_name}")
            else:
                print("Embedding model failed to load.")
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")
            cls.embeddings = None

    @classmethod
    def generate_vectors_from_documents(cls, chunks=None):
        """
        Generates FAISS vector store from document chunks.
        """
        try:
            if cls.embeddings is None:
                print("Embedding model not initialized.")
                return None
            if not chunks:
                print("No chunks provided for vector generation.")
                return None
            return FAISS.from_documents(chunks, embedding=cls.embeddings, normalize_L2=True)
        except Exception as e:
            print(f"Error generating vectors: {e}")
            return None


# ========================== PREPARE TEXT ============================

class PrepareText:
    """
    Reads, cleans, chunks, and vectorizes PDF text for Gemini QA pipeline.
    """

    def __init__(self, file_path: str, config: GeminiConfig, api_key: str = None):
        try:
            self.file_path = file_path
            self.config = config
            self.api_key = api_key
            self.raw_text = ReadFile.read_pdf_file(file_path)
            if self.raw_text:
                print(f"Successfully read PDF: {file_path}")
            else:
                print(f"PDF is empty or could not be read: {file_path}")
        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.raw_text = ""

    def clean_data(self) -> str:
        """
        Cleans text for embeddings (lowercase, remove punctuation & noise).
        """
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

    def get_chunks(self, separator=None, chunksize=1000, overlap=100):
        """
        Splits cleaned text into structured document chunks.
        """
        try:
            TextChunks.initialize(separator=separator, chunksize=chunksize, overlap=overlap)
            chunks = TextChunks.get_text_chunks_doc(text=self.clean_data())
            print(f"Created {len(chunks)} text chunks.")
            return chunks
        except Exception as e:
            print(f"Error creating chunks: {e}")
            return []

    def create_text_vectors(self, separator=None, chunksize=None, overlap=None):
        """
        Generates FAISS vector store for similarity search.
        """
        try:
            Vectors.initialize(config=self.config)
            vectors = Vectors.generate_vectors_from_documents(
                chunks=self.get_chunks(separator, chunksize, overlap)
            )
            if vectors:
                print("Vector store successfully created.")
            else:
                print("Vector store creation failed.")
            return vectors
        except Exception as e:
            print(f"Error creating vectors: {e}")
            return None


if __name__ == "__main__":
    pass
