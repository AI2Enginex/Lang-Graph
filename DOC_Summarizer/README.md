# Lang-Graph Document Summarization Project 

This project leverages LangGraph and LangChain to perform efficient and accurate summarization of large documents. By utilizing advanced language models and structured processing techniques, the system provides concise summaries, enhancing information retrieval and comprehension.

## Overview
***The Lang-Graph Document Summarization Project is designed to process and summarize extensive documents using two primary strategies***:

1. Stuff Summarization: Processes the entire document in a single pass to generate a summary.
   
2. Map-Reduce Summarization: Breaks the document into chunks, summarizes each chunk, and then combines these summaries into a cohesive final summary.
By employing these methods, the system ensures flexibility and effectiveness in handling documents of varying lengths and complexities.

## Detailed Workflow

## 1. Text Preparation
  ***The process begins with the PrepareText class, responsible for reading and preprocessing the document**:
   
        Reading: Loads the document content.
         
        Cleaning: Removes unnecessary characters, extra spaces, and converts text to lowercase.
         
        Chunking: Splits the text into manageable chunks based on the specified delimiter, chunk size, and overlap.
         
        Vectorization: Transforms text chunks into vector representations using the specified embedding model.
   
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
   
  ***This preparation ensures the text is in an optimal state for summarization***.


## 2. Summarization Strategies
   **The project implements two primary summarization strategies**:

   ### a. Stuff Summarization
   
   ***In this approach, the entire document is processed in a single pass:***

      Initialization: The StuffSummarizer class inherits from both PrepareText and ChatGoogleGENAI. It initializes by preparing the text and setting up the language model.
      
      Summarization: The entire cleaned text is fed into the language model to generate a concise summary.

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
      
   ***This method is suitable for shorter documents where the entire content can be processed at once.***
   
   ### b. Map-Reduce Summarization

   ***For larger documents, a more structured approach is employed***:

      Mapping: The document is divided into chunks, and each chunk is summarized individually.
   
      Reducing: The individual summaries are then combined to form a comprehensive summary of the entire document.

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
   ***This strategy ensures that even extensive documents are summarized effectively without losing context***.

## 3. Graph Execution

   ***The project utilizes LangGraph to define and execute the summarization workflows***:

      Graph Definition: Nodes represent different processing steps (e.g., retrieving chunks, summarizing). Edges define the flow between these nodes.
      
      State Management: The state of the document (e.g., current chunk, partial summaries) is maintained throughout the process.
      
      Execution: The graph is executed, ensuring that each step is performed in the correct sequence with proper state management.

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
      
   ***This structured approach allows for flexible and efficient processing, accommodating various document sizes and complexities***.
