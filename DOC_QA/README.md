## 📘 Document Question Answering using LangGraph

## 🚀 Overview

This project implements a Document Question Answering (QA) system using LangGraph, Google Gemini API, FAISS, and LangChain. It allows users to query a PDF document and retrieve accurate, context-aware answers.

## 🌟 Features

📄 PDF Document Processing (Load, Split, Embed)

🔎 Semantic Search using FAISS

🤖 Question Answering powered by Google Gemini API

🧠 Chain-of-Thought (CoT) Reasoning for step-by-step answers

✅ Verification Prompting to ensure answer correctness

🔄 Modular & Extensible Code with Clean Class-based Design

## 🛠 How It Works

 **1️. Load and Index the Document**

  Extracts text from a PDF file.
    
  Splits text into chunks using RecursiveCharacterTextSplitter.
    
  Embeds and stores chunks in FAISS.

  Uses similarity search (k=4) to retrieve top-matching chunks from FAISS.
    
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

    
    # Defining a QAsystem class for Question and Answering 
    # the class purpose is to retrieve the chunks and
    # answer based on the selected Prompt Template
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
        
 **2. Select Prompt Type**

  ***The user can choose one of the following prompting strategies:***

  RAG (Keyword Extraction): Ensures the answer is strictly based on the document.
    
  Chain-of-Thought (CoT): Guides the model to think step-by-step.
    
  Verification: Checks if the answer is fully supported by the document.

    ## The given Prompt-Templates are focused for Question and Anaswering
    class PromptTemplates:
        """
        This class provides reusable prompt templates for different prompting strategies:
        1. Keyword Extraction (RAG style factual QA)
        2. Chain-of-Thought reasoning (step-by-step logical answering)
        3. Verification prompts (double-check factual correctness)
        """


  **3. Graph-Based Execution Flow:**

  ***When QASystemGraphExecuetion is instantiated and executed:***

  Document Processing:

    The document is split into chunks based on the given delimiter and chunk size.
    
    Each chunk is embedded using the specified model (e.g., Google Generative AI embeddings).
    
    The embeddings are stored in FAISS.
  
  Graph-Based Execution Flow:
  
    Retrieval Node: Finds the most relevant document chunks using similarity search.
    
    Answer Generation Node: Chooses the appropriate prompt template (RAG, CoT, Verification). Calls the Google Gemini API with the retrieved chunks and selected prompt.
  
  Returns Answer:
  
    The system generates and returns an answer based strictly on the document content.

    # Class for QASystem Execuetion
    # creating a Graph Execuetion flow
    class QASystemGraphExecuetion(QASystem):

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
            - Node 2: Fetch the Question and provide the answers.
            - Edge: Connects retrieve → QA.
            """
