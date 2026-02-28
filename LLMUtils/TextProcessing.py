from LLMUtils.ReadData import ReadFile, ReadExcel
import re
import os
import pandas as pd
from langchain_core.documents import Document
from LLMUtils.PrepareChunks import TextChunks
from LLMUtils.VectoreStores import Vectors
import cleantext

# ========================== PREPARE TEXT ============================

class PrepareText:
    """
    Reads, cleans, chunks, and vectorizes MULTIPLE PDF files
    with section detection metadata.
    """

    def __init__(self, file_paths, config=None, api_key: str = None):
        try:
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            self.file_paths = file_paths
            self.config = config
            self.api_key = api_key

            self.all_pages = []

            for file_path in self.file_paths:
                pages = ReadFile.read_pdf_pages(file_path=file_path)
                for page in pages:
                    page["source"] = file_path
                self.all_pages.extend(pages)

            print(f"Total pages loaded from all PDFs: {len(self.all_pages)}")

        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.all_pages = []

    def clean_data(self, text):
        try:
            return cleantext.clean(
                text,
                lowercase=False,
                punct=False,
                extra_spaces=True
            )
        except Exception:
            return text

    def detect_section(self, text):
        """
        Extracts all possible section headings from a page.
        Returns list of detected headings in order.
        """
        try:
            section_patterns = [
                r'^\d+(\.\d+)+\s+[A-Z][A-Z0-9\s\-\(\)&]+$',
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

            return sections if sections else None

        except Exception as e:
            print(f"Error detecting section: {e}")
            return None

    def get_chunks(self, chunk_size=1200, overlap=200, separator=None):

        splitter = TextChunks.initialize(
            separator=separator,
            chunksize=chunk_size,
            overlap=overlap
        )

        final_docs = []

        for page in self.all_pages:

            original_text = page["text"]
            cleaned_text = self.clean_data(original_text)
            detected_sections = self.detect_section(original_text)

            splits = splitter.split_text(cleaned_text)

            for j, chunk in enumerate(splits):

                metadata = {
                    "page": page["page_number"],
                    "source": page["source"],
                    "file_name": os.path.basename(page["source"]),
                    "section": detected_sections,
                    "chunk_id": f"{page['source']}_p{page['page_number']}_c{j}"
                }

                final_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )

        print(f"Created {len(final_docs)} chunks from multiple PDFs.")
        return final_docs

    def create_text_vectors(self, separator=None, chunksize=1000, overlap=100):

        Vectors.initialize(config=self.config)

        chunks = self.get_chunks(
            chunk_size=chunksize,
            overlap=overlap,
            separator=separator
        )
        
        for doc in chunks:print(doc.metadata)
        vectors = Vectors.generate_vectors_from_documents(chunks=chunks)

        if vectors:
            print("Vector store successfully created for multiple PDFs.")
        else:
            print("Vector store creation failed.")

        return vectors




# ========================== PREPARE Excel ============================

class PrepareExcel:
    """
    Reads, cleans, chunks, and vectorizes MULTIPLE Excel files.
    """

    def __init__(self, file_paths, config=None, api_key: str = None):

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self.file_paths = file_paths
        self.config = config
        self.api_key = api_key

        self.raw_text = ""
        self.process_all_excels()

    def process_all_excels(self):

        all_text = []

        for file_path in self.file_paths:

            try:
                if file_path.endswith(".xls"):
                    sheets = pd.read_excel(file_path, sheet_name=None, engine="xlrd")
                else:
                    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

                for sheet_name, df in sheets.items():

                    all_text.append(f"\n\n===== FILE: {file_path} | SHEET: {sheet_name} =====\n")

                    text_columns = df.select_dtypes(include=["object"]).columns

                    for row_index, row in df.iterrows():
                        row_parts = []

                        for col in text_columns:
                            value = str(row[col]) if pd.notna(row[col]) else ""
                            row_parts.append(f"{col}: {value}")

                        if row_parts:
                            formatted_row = f"[Row {row_index}] " + " | ".join(row_parts)
                            all_text.append(formatted_row)

            except Exception as e:
                print(f"Error processing Excel {file_path}: {e}")

        self.raw_text = "\n".join(all_text)
        print("All Excel files processed.")

    def clean_data(self):
        return cleantext.clean(
            self.raw_text,
            lowercase=True,
            punct=True,
            extra_spaces=True
        )

    def get_chunks(self, separator=None, chunksize=1000, overlap=100):

        splitter = TextChunks.initialize(
            separator=separator,
            chunksize=chunksize,
            overlap=overlap
        )

        splits = splitter.split_text(self.clean_data())

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": "multiple_excel_files",
                    "chunk_id": f"excel_chunk_{i}"
                }
            )
            for i, chunk in enumerate(splits)
        ]

        print(f"Created {len(documents)} chunks from multiple Excel files.")
        return documents

    def create_text_vectors(self, separator=None, chunksize=1000, overlap=100):

        Vectors.initialize(config=self.config)

        vectors = Vectors.generate_vectors_from_documents(
            chunks=self.get_chunks(separator, chunksize, overlap)
        )

        if vectors:
            print("Vector store successfully created for multiple Excel files.")
        else:
            print("Vector store creation failed.")

        return vectors

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
        api_key=api_key
    )

    file_paths = [
        "E:/Lang-Graph/Book.pdf",
        "E:/Lang-Graph/TATAAGM.pdf",
    ]

    # Process PDFs
    text_processor = PrepareText(file_paths=file_paths, config=config, api_key=api_key)
    pdf_vectors = text_processor.create_text_vectors(
        chunksize=1500,
        overlap=250,
        separator=["\n\n", "\n", " ", ""]
    )

    print(pdf_vectors)