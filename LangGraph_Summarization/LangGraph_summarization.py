from langgraph.graph import StateGraph
from LLMUtils.PromptClass import PromptTemplates
from LLMUtils.TextProcessing import PrepareText
from LLMUtils.LLMConfigs import GeminiConfig, ChatGoogleGENAI, SimpleDocState, ReducedDocState, api_key

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

    def __init__(self, filename=None, delimiter=None, chunk=None, over_lap=None, config=None):
        # initialize PrepareText with filename
        PrepareText.__init__(self, file_path=filename,config=config)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self,config=config)

        # calling the preprocessed vectors
        self.vector_store = self.create_text_vectors(
            separator=delimiter,           
            chunksize=chunk, 
            overlap=over_lap
        )

        # Create FAISS  as Retriever
        self.retriever = None
        if self.vector_store:
                self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 6}
            )
    
    # function to retrieve the chunks
    def retrieve_chunks(self, state: SimpleDocState):
        try:
            query = state["messages"][-1]
            results = self.retriever.invoke(query)
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

    def __init__(self, filename=None, delimiter=None, chunk=None, over_lap=None, config=None):
        # initialize PrepareText with filename
        PrepareText.__init__(self, file_path=filename,config=config)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self,config=config)

        # Create vector store by chunking and embedding the document
        self.vector_store = self.create_text_vectors(
            separator=delimiter,
            chunksize=chunk,
            overlap=over_lap
        )

        # Create FAISS  as Retriever
        self.retriever = None
        if self.vector_store:
                self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 6}
            )
    
    # function to retrieve chunks
    def retrieve_chunks(self, state: ReducedDocState):
        try:
            query = state["messages"][-1]  # Last message is query string
            results = self.retriever.invoke(query)

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

    def __init__(self, data=None, processing_delimiter=None, total_chunk=None, overlapping=None,config=None):
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
        super().__init__(filename=data, delimiter=processing_delimiter, chunk=total_chunk, over_lap=overlapping, config=config)

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

    def __init__(self, data=None, processing_delimiter=None, total_chunk=None, overlapping=None, config=None):
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
        super().__init__(filename=data, delimiter=processing_delimiter, chunk=total_chunk, over_lap=overlapping, config=config)
    
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
        
        
# ========================== SUMMARIZER MANAGER ==========================
class SummarizerManager:
    """
    Manages document summarization using different chain types.
    """

    def __init__(self, file_path: str, separator, chunk_size: int, overlap: int, config: None):
        self.file_path = file_path
        self.separator = separator
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.config = config

    def get_summarizer(self, chain_type: str):
        """
        Returns the appropriate summarizer object based on chain_type.
        """
        try:
            if chain_type.lower() == "simple":
                return StuffGraphExecuetion(
                    data=self.file_path,
                    processing_delimiter=self.separator,
                    total_chunk=self.chunk_size,
                    overlapping=self.overlap,
                    config=config
                )
            else:
                return MapReduceGraphExecuetion(
                    data=self.file_path,
                    processing_delimiter=self.separator,
                    total_chunk=self.chunk_size,
                    overlapping=self.overlap,
                    config=config
                )
        except Exception as e:
            print(f"Error initializing summarizer for chain type '{chain_type}': {e}")
            return None

    def summarize(self, chain_type: str, query: str) -> str:
        """
        Summarize the document using the selected chain type.
        """
        try:
            summarizer = self.get_summarizer(chain_type)
            if summarizer is None:
                return "Failed to initialize summarizer."
            return summarizer.summarize(query=query)
        except Exception as e:
            print(f"Error during summarization: {e}")
            return ""

# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    try:

        config = GeminiConfig(
            chat_model_name="gemini-2.5-flash",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0.1,
            top_p=0.8,
            top_k=32,
            max_output_tokens=3500,
            generation_max_tokens=8192,
            api_key=api_key  # Set your key here or via environment variable
        )
        # File path and parameters
        file_path = "E:/Lang-Graph/wings_of_fire.pdf"
        separator = ["\n\n", "\n", " ", ""]
        chunk_size = 3500
        overlap = 100

        # User input
        chain_type = input("Enter chain type (simple / other): ")
        query = "Summarize this document briefly."



        # Initialize summarizer manager
        manager = SummarizerManager(
            file_path=file_path,
            separator=separator,
            chunk_size=chunk_size,
            overlap=overlap,
            config=config
        )

        # Generate summary
        summary = manager.summarize(chain_type=chain_type, query=query)
        print("Summary:\n", summary)

    except Exception as e:
        print(f"Error in main execution: {e}")