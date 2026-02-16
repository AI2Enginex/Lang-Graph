from PyPDF2 import PdfReader
from textwrap import dedent
import re
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from LLMUtils.LLMConfigs import  EmbeddingModel
import cleantext


# ========================== FILE UTILITIES ============================

class ReadFile:
    """
    Utility to read text from PDFs or plain text files.
    """

    @classmethod
    def read_pdf_file(cls, file_path: str):
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
        
    @classmethod
    def read_pdf_pages(cls, file_path: str):
        """
        Reads PDF page-wise and preserves page numbers.
        """
        try:
            reader = PdfReader(file_path)
            pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({
                        "page_number": i + 1,
                        "text": text
                    })

            print(f"Loaded {len(pages)} pages.")
            return pages

        except Exception as e:
            print(f"Error reading PDF file '{file_path}': {e}")
            return []

# Class for Reading Data from
# an Excel Sheets and creating 
# VectoreStores
class ReadExcel:
    """
    Utility class for processing Excel files
    """

    def extract_text_columns(state):
        try:
            df = state["dataframe"]

            text_columns = df.select_dtypes(include=["object"]).columns

            documents = []
            for col in text_columns:
                documents.extend(df[col].dropna().astype(str).tolist())

            state["documents"] = documents
            return state
        
        except Exception as e:
            return e

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
            return cls.text_splitter
        except Exception as e:
            print(f"Failed to initialize text splitter: {e}")
            cls.text_splitter = None



# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config=None):
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
    Reads, cleans, chunks, and vectorizes PDF text with metadata.
    """

    def __init__(self, file_path: str, config=None, api_key: str = None):
        try:
            self.file_path = file_path
            self.config = config
            self.api_key = api_key

            # Page-wise loading
            self.pages = ReadFile.read_pdf_pages(file_path=file_path)

            if self.pages:
                print(f"Successfully read PDF: {file_path}")
            else:
                print(f"PDF is empty or unreadable: {file_path}")

        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.pages = []

    def clean_data(self, text):
        try:
            return cleantext.clean(
                text,
                lowercase=False,
                punct=False,
                extra_spaces=True
            )
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

    def detect_section(self, text):
        
        try:
            
            """
            Extracts all possible section headings from a page.
            Returns list of detected headings in order.
            """
            section_patterns = [
                r'^\d+\.?\s+[A-Z][A-Z0-9\s\-\(\)&]+$',
                r'^Clause\s+\d+.*',            
                r'^[A-Z][A-Z0-9\s\-\(\)&]{5,}$'
            ]

            sections = []
            lines = text.split("\n")

            for line in lines:
                line = line.strip()

                for pattern in section_patterns:
                    if re.match(pattern, line):
                        sections.append(line)
                        break

            return sections
        except Exception as e:
            return e

    def get_chunks(self, chunk_size=1200, overlap=200, separator=None):
        
        try:
            splitter = TextChunks.initialize(separator=separator, chunksize=chunk_size, overlap=overlap)

            final_docs = []

            for page in self.pages:

                page_text = page.get("text", "")
                if not page_text:
                    continue

                cleaned_text = self.clean_data(page_text)
                section = self.detect_section(page_text)

                splits = splitter.split_text(cleaned_text)

                for j, chunk in enumerate(splits):

                    metadata = {
                        "page": page["page_number"],
                        "source": self.file_path,
                        "section": section,
                        "chunk_id": f"{self.file_path}_p{page['page_number']}_c{j}"
                    }

                    final_docs.append(
                        Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                    )

            print(f"Created {len(final_docs)} chunks with metadata.")
            
            return final_docs
        except Exception as e:
            return e

    def create_text_vectors(self, separator=None, chunksize=1000, overlap=100):

        try:
            Vectors.initialize(config=self.config)

            chunks = self.get_chunks(
                chunk_size=chunksize,
                overlap=overlap,
                separator=separator
            )

            print("Creating Vector Chunks")

            for doc in chunks:print(doc.metadata)

            vectors = Vectors.generate_vectors_from_documents(chunks=chunks)

            if vectors:
                print("Vector store successfully created.")
            else:
                print("Vector store creation failed.")

            return vectors

        except Exception as e:
            print(f"Error creating vectors: {e}")
            return None




# ========================== PREPARE Excel ============================

class PrepareExcel:
    """
    Reads, cleans, chunks, and vectorizes Excel text data
    (supports multiple sheets) for Gemini QA pipeline.
    """

    def __init__(self, file_path: str, config=None, api_key: str = None):
        try:
            self.file_path = file_path
            self.config = config
            self.api_key = api_key

            # Load all sheets
            if file_path.endswith(".xls"):
                self.sheets = pd.read_excel(file_path, sheet_name=None, engine="xlrd")
            else:
                self.sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

            if self.sheets:
                print(f"Successfully read Excel file with {len(self.sheets)} sheets: {file_path}")
            else:
                print(f"Excel file is empty: {file_path}")

            # Convert all sheets into one structured string
            self.raw_text = self.convert_excel_to_text()

        except Exception as e:
            print(f"Error initializing PrepareExcel: {e}")
            self.sheets = {}
            self.raw_text = ""


    def convert_excel_to_text(self):
        """
        Converts text columns of all sheets into structured text format
        while clearly preserving sheet boundaries.
        """
        try:
            if not self.sheets:
                return ""

            all_text = []

            for sheet_name, df in self.sheets.items():

                # Clear sheet boundary marker
                all_text.append(f"\n\n========== SHEET START: {sheet_name} ==========\n")

                text_columns = df.select_dtypes(include=["object"]).columns

                if len(text_columns) == 0:
                    print(f"No textual columns found in sheet: {sheet_name}")
                    continue

                for row_index, row in df.iterrows():

                    row_parts = []

                    for col in text_columns:
                        value = str(row[col]) if pd.notna(row[col]) else ""
                        row_parts.append(f"{col}: {value}")

                    if row_parts:
                        # Explicit row separation
                        formatted_row = f"[Row {row_index}] " + " | ".join(row_parts)
                        all_text.append(formatted_row)

                # Clear sheet end marker
                all_text.append(f"\n========== SHEET END: {sheet_name} ==========\n")

            return "\n".join(all_text)

        except Exception as e:
            print(f"Error converting Excel to text: {e}")
            return ""


    def clean_data(self):
        """
        Cleans text for embeddings.
        """
        try:
            return cleantext.clean(
                self.raw_text,
                lowercase=True,
                punct=True,
                extra_spaces=True
            )
        except Exception as e:
            print(f"Error cleaning Excel data: {e}")
            return self.raw_text


    def get_chunks(self, separator=None, chunksize=1000, overlap=100):
        """
        Splits cleaned Excel text into document chunks.
        """
        try:
            TextChunks.initialize(
                separator=separator,
                chunksize=chunksize,
                overlap=overlap
            )

            chunks = TextChunks.get_text_chunks_doc(
                text=self.clean_data()
            )

            print(f"Created {len(chunks)} Excel text chunks.")
            print(chunks)
            return chunks

        except Exception as e:
            print(f"Error creating Excel chunks: {e}")
            return []


    def create_text_vectors(self, separator=None, chunksize=None, overlap=None):
        """
        Generates FAISS vector store for Excel semantic search.
        """
        try:
            Vectors.initialize(config=self.config)

            vectors = Vectors.generate_vectors_from_documents(
                chunks=self.get_chunks(separator, chunksize, overlap)
            )

            if vectors:
                print("Excel vector store successfully created.")
            else:
                print("Excel vector store creation failed.")

            return vectors

        except Exception as e:
            print(f"Error creating Excel vectors: {e}")
            return None

if __name__ == "__main__":

    from LLMUtils.LLMConfigs import ChatGoogleGENAI, GeminiConfig, QAState, api_key
    
    config = GeminiConfig(
            chat_model_name="gemini-3-flash-preview",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0,
            top_p=0.8,
            top_k=32,
            max_output_tokens=3000,
            generation_max_tokens=8192,
            api_key=api_key  # Set your key here or via environment variable
        )

    file_path = "E:/Lang-Graph/Book.pdf"
    text = PrepareText(file_path=file_path, config=config,api_key=api_key)
    data = text.create_text_vectors(chunksize=1200, overlap=250, separator=["\n\n", "\n", " ", ""])
    print(data)