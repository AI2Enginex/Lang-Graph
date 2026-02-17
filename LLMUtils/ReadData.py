from PyPDF2 import PdfReader
import pandas as pd

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

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            list: return a list of dictionaries with page number and text.
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

class ReadExcel:
    """
    Utility class for processing Excel files
    """
    
    @classmethod
    def read_excel_files(cls, file: str):
        try:
            if file.endswith(".xls"):
                sheets = pd.read_excel(file, sheet_name=None, engine="xlrd")
            else:
                sheets = pd.read_excel(file, sheet_name=None, engine="openpyxl")

            if sheets:
                print(f"Successfully read Excel file with {len(sheets)} sheets: {file}")
            else:
                print(f"Excel file is empty: {file}")
            
            return sheets
        
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return {}
