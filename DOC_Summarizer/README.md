## Lang-Graph Document Summarization Project

This project leverages LangGraph and LangChain to perform efficient and accurate summarization of large documents. By utilizing advanced language models and structured processing techniques, the system provides concise summaries, enhancing information retrieval and comprehension.

## Overview
The Lang-Graph Document Summarization Project is designed to process and summarize extensive documents using two primary strategies:

1. Stuff Summarization: Processes the entire document in a single pass to generate a summary.
   
2. Map-Reduce Summarization: Breaks the document into chunks, summarizes each chunk, and then combines these summaries into a cohesive final summary.
By employing these methods, the system ensures flexibility and effectiveness in handling documents of varying lengths and complexities.

## Detailed Workflow

## 1. Text Preparation
    The process begins with the PrepareText class, responsible for reading and preprocessing the document:
   
    Reading: Loads the document content.
   
    Cleaning: Removes unnecessary characters, extra spaces, and converts text to lowercase.
   
    Chunking: Splits the text into manageable chunks based on the specified delimiter, chunk size, and overlap.
   
    Vectorization: Transforms text chunks into vector representations using the specified embedding model.
   
    This preparation ensures the text is in an optimal state for summarization.
