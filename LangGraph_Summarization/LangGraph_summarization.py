from langgraph.graph import StateGraph, END
from LLMUtils.PromptClass import PromptTemplates
from LLMUtils.TextProcessing import PrepareText
from LLMUtils.LLMConfigs import GeminiConfig, ChatGoogleGENAI, AgentDocState,api_key

# helper function
# to extract text
# from a list
def extract_text(content):
    if isinstance(content, list): # checf if the parameter is a list
        return " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)


class StuffSummarizer(PrepareText, ChatGoogleGENAI):
    """
    Tool for simple (stuff) summarization.
    Retrieves chunks and summarizes them in one pass.
    """

    def __init__(self, filename=None, delimiter=None, chunk=None, over_lap=None, config=None):

        # Load document and embeddings
        PrepareText.__init__(self, file_path=filename, config=config)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self, config=config)

        # calling the preprocessed vectors
        self.vector_store = self.create_text_vectors(
            separator=delimiter,
            chunksize=chunk,
            overlap=over_lap
        )
        
        # create FAISS as Retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
    
    # function to retrieve the chunks
    def retrieve_chunks(self, state: AgentDocState):
        """
        Retrieve relevant document chunks.
        ALWAYS returns full AgentDocState.
        """
        try:
            query = state["messages"][-1]   # reads the message key from the Doc State
            docs = self.retriever.invoke(query)

            return {
                **state,
                "document_chunks": docs,
                "partial_summaries": []
            }

        except Exception as e:
            return {
                **state,
                "reasoning": f"Retrieval failed: {str(e)}",
                "document_chunks": [],
                "partial_summaries": []
            }

    def summarize_chunks(self, state: AgentDocState):
        """
        Summarize the entire document in one pass.
        """
        try:

            # Implementing the logic for Stuff Summarizer
            # Use ALL chunks, not retrieved chunks
            content = "\n\n".join(
                doc.page_content for doc in self.vector_store.docstore._dict.values()
            )

            prompt = PromptTemplates.summarisation_prompt().format(context=content)
            response = self.llm.invoke(prompt)

            return {
                **state,
                "messages": state["messages"] + [response.content]
            }

        except Exception as e:
            return {
                **state,
                "reasoning": f"Stuff summarization failed: {str(e)}"
            }


class MapReduceSummarizer(PrepareText, ChatGoogleGENAI):
    """
    Tool for MapReduce-style summarization.
    Used for long or complex documents.
    """

    def map_summarize(self, state: AgentDocState):
        """
        Map step: summarize each chunk individually.
        """
        try:

            # creating a list to store
            # the partial summaries
            # for each Document chunk
            summaries = []   
            prompt_template = PromptTemplates.map_prompt()
            

            # summarizing each chunk
            for chunk in state["document_chunks"]:
                prompt = prompt_template.format(context=chunk.page_content)
                result = self.llm.invoke(prompt)
                summaries.append(result.content)

            return {
                **state,
                "partial_summaries": summaries
            }

        except Exception as e:
            return {
                **state,
                "reasoning": f"Map step failed: {str(e)}",
                "partial_summaries": []
            }

    def reduce_summarize(self, state: AgentDocState):
        """
        Reduce step: combine partial summaries into final summary.
        Always normalizes content to string.
        """
        try:
            # normalize partial summaries
            normalized_partials = []
            for item in state["partial_summaries"]:
                if isinstance(item, list):
                    normalized_partials.append(
                        " ".join(
                            elem.get("text", "")
                            for elem in item
                            if isinstance(elem, dict)
                        )
                    )
                else:
                    normalized_partials.append(str(item))

            combined = "\n\n".join(normalized_partials)

            prompt = PromptTemplates.reduce_prompt().format(context=combined)
            result = self.llm.invoke(prompt)

            # normalize final summary
            final_text = (
                " ".join(elem.get("text", "") for elem in result.content)
                if isinstance(result.content, list)
                else str(result.content)
            )

            return {
                **state,
                "messages": state["messages"] + [final_text]
            }

        except Exception as e:
            return {
                **state,
                "reasoning": f"Reduce step failed: {str(e)}"
            }


# class for Implementing
# the logic for the Agent
class AgenticSummarizer(StuffSummarizer, MapReduceSummarizer):
    """
    Agent brain responsible for:
    1. Deterministic strategy selection
    2. Reflection on summary quality (LLM)
    3. Optional refinement (LLM)
    """

    # STRATEGY DECISION
    def decide_strategy(self, state: AgentDocState):
        """
        Decide whether to use 'stuff' or 'map_reduce' summarization.

        This is CONTROL FLOW → must be deterministic.
        """

        try:
            chunk_count = len(self.vector_store.index_to_docstore_id)
            
            # Defining the strategy for Prompt selection
            strategy = "stuff" if chunk_count <= 90 else "map_reduce"

            return {
                **state,
                "strategy": strategy,
                "needs_refinement": False,
                "reasoning": f"Strategy '{strategy}' selected because chunk_count={chunk_count}"
            }
        except Exception as e:
            return {
                **state,
                "needs_refinement": False,
                "reasoning": f"{str(e)}"
            }



    # REFLECTION (LLM)
    # function to decide whether the
    # generates summaries are correct or 
    # needs any correction
    def reflect_summary(self, state: AgentDocState):
        """
        Decide whether the generated summary needs refinement.
        """

        try:

            # The generated Summary is again passed to LLM
            # to check whether the generated summary needs refinement
            summary = state["messages"][-1]

            prompt = PromptTemplates.reflect_prompt_template(summary=summary)

            response = self.llm.invoke(prompt)  # The LLM is used to make the decision
            text = extract_text(response.content).strip().upper()

            needs_refinement = "NO" not in text

            return {
                **state,
                "needs_refinement": needs_refinement,
                "reasoning": "Reflection completed successfully"
            }

        except Exception as e:
            # Reflection failure must NOT break the graph
            return {
                **state,
                "needs_refinement": False,
                "reasoning": f"Reflection skipped due to error: {str(e)}"
            }

  
    # REFINEMENT
    # function to refine the generated summary
    # the llm tries to resummarize the content
    def refine_summary(self, state: AgentDocState):
        """
        Improve the summary if reflection indicates refinement is needed.
        """

        try:
            summary = state["messages"][-1]

            prompt = PromptTemplates.refine_prompt_template(summary=summary)

            improved = self.llm.invoke(prompt)

            return {
                **state,
                "messages": state["messages"] + [improved.content],
                "reasoning": "Summary refined successfully"
            }

        except Exception as e:
            # Refinement failure should still return state
            return {
                **state,
                "reasoning": f"Refinement failed: {str(e)}"
            }

class AgenticGraphExecution(AgenticSummarizer):
    """
    AgenticGraphExecution builds and executes a LangGraph-based
    Agentic RAG summarization workflow.

    """

    def __init__(self,data=None,processing_delimiter=None,total_chunk=None,overlapping=None,config=None):
        super().__init__(
            filename=data,
            delimiter=processing_delimiter,
            chunk=total_chunk,
            over_lap=overlapping,
            config=config
        )

   
    # Routing functions
    @staticmethod
    def route_strategy(state):
        return "stuff" if state["strategy"] == "stuff" else "map_reduce"


    @staticmethod
    def route_refinement(state: AgentDocState):
        return "refine" if state["needs_refinement"] else END

 
    # Build Agentic Graph
    def build_graph(self):
        graph = StateGraph(AgentDocState)

        # Agent reasoning nodes
        graph.add_node("decide", self.decide_strategy)
        graph.add_node("reflect", self.reflect_summary)
        graph.add_node("refine", self.refine_summary)

        graph.add_node("retrieve", self.retrieve_chunks)
        graph.add_node("stuff", self.summarize_chunks)
        graph.add_node("map", self.map_summarize)
        graph.add_node("reduce", self.reduce_summarize)

        # Entry point
        graph.set_entry_point("decide")

        # Strategy routing
        graph.add_conditional_edges(
            "decide",
            self.route_strategy,
            {
                "stuff": "stuff",
                "map_reduce": "retrieve"
            }
        )

        # MapReduce path
        graph.add_edge("retrieve", "map")
        graph.add_edge("map", "reduce")
        graph.add_edge("reduce", "reflect")

        # Stuff path
        graph.add_edge("stuff", "reflect")

        # Reflection routing
        graph.add_conditional_edges(
            "reflect",
            self.route_refinement,
            {
                "refine": "refine",
                END: END
            }
        )

        graph.add_edge("refine", END)

        return graph

    def summarize(self, query: str):
        graph_executor = self.build_graph().compile()

        initial_state = {
            "messages": [query],
            "document_chunks": [],
            "partial_summaries": [],
            "strategy": "",
            "needs_refinement": False,
            "reasoning": ""
        }

        final_state = graph_executor.invoke(initial_state)

        final_summary = extract_text(final_state["messages"][-1])

        return {
            "summary": final_summary,
            "strategy_used": final_state["strategy"],
            "refinement_status": final_state["needs_refinement"],
            "reasoning": final_state["reasoning"]
            
        }

    
# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    
   
        # LLM + Embedding Configuration
        config = GeminiConfig(
            chat_model_name="gemini-3-flash-preview",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0.1,
            top_p=0.8,
            top_k=32,
            max_output_tokens=3500,
            generation_max_tokens=8192,
            api_key=api_key
        )

 
        # File & Chunking Parameters
        file_path = "E:/Lang-Graph/wings_of_fire.pdf"
        separator = ["\n\n", "\n", " ", ""]
        chunk_size = 3500
        overlap = 100


        # User Query
        query = "Summarize this document briefly."

  
        # Initialize Agentic Graph
        agent = AgenticGraphExecution(
            data=file_path,
            processing_delimiter=separator,
            total_chunk=chunk_size,
            overlapping=overlap,
            config=config
        )

      
        # Execute Agentic Summarization
        result = agent.summarize(query=query)

        # Output
        print("\n================= SUMMARY =================\n")
        print(result["summary"])

        print("\n============= AGENT DECISION =============\n")
        print("Strategy Used :", result["strategy_used"])
        print("Needs Refinement :", result["refinement_status"])
        print("Agent Reason  :", result["reasoning"])

    
