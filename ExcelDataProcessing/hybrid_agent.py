from langgraph.graph import StateGraph
from LLMUtils.TextProcessing import PrepareExcel
from LLMUtils.LLMConfigs import GeminiConfig, ChatGoogleGENAI, ExcelAgentState, api_key


class HybridExcelAgent(ChatGoogleGENAI, PrepareExcel):

    def __init__(self, filename: str, config=None,separator=None, chunk_size=None, over_lap=None):
        
        # Load document and embeddings
        PrepareExcel.__init__(self, file_path=filename, config=config)

        # initialize ChatGoogleGENAI 
        ChatGoogleGENAI.__init__(self, config=config)
        
        # create FAISS as Retriever
        self.retriever = self.create_text_vectors(separator=separator,chunksize=chunk_size,overlap=over_lap)

    def route_intent(self, state):
        try:
            prompt = f"""
            Classify the question into:
            - semantic
            - analytical
            - hybrid

            Question: {state["question"]}
            """

            response = self.llm.invoke(prompt)
            state["intent"] = response.content.lower().strip()

            return state
        except Exception as e:
            
            state["error"] = str(e)
            return state


    # RAG Node
    def rag_node(self, state):

        try:
            docs = self.retriever.invoke(state["question"])

            state["retrieved_docs"] = "\n".join(
                [doc.page_content for doc in docs]
            )

            return state
        except Exception as e:

            state["error"] = str(e)
            return state
         

 
    # Pandas Tool Node
    def pandas_tool(self, state):

        try:
            df = state["dataframe"]

            prompt = f"""
            Convert this question into valid pandas code.
            DataFrame name is df.
            Only return executable Python code.

            Question: {state["question"]}
            """

            response = self.llm.invoke(prompt)
            code = response.content.strip()

            local_vars = {"df": df}

            try:
                exec(code, {}, local_vars)
                state["tool_output"] = str(local_vars)
            except Exception as e:
                state["tool_output"] = f"Execution error: {e}"

            return state
        
        except Exception as e:
            state["error"] = str(e)
            return state
        

    # Final Answer Node
    def generate_final_answer(self, state):

        try:
            prompt = f"""
            Provide a clear final answer.

            Retrieved Context:
            {state.get("retrieved_docs", "")}

            Analytical Result:
            {state.get("tool_output", "")}

            Question:
            {state["question"]}
            """

            response = self.llm.invoke(prompt)
            state["final_answer"] = response.content

            return state
        except Exception as e:
            state["error"] = str(e)
            return state
        
if __name__ == "__main__":

    config = GeminiConfig(
        chat_model_name="gemini-3-flash-preview",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        temperature=0.0,
        top_p=0.8,
        top_k=32,
        max_output_tokens=3000,
        generation_max_tokens=8192,
        api_key=api_key
    )

    file_path = "E:/Lang-Graph/excel_data_file.xls"

    agent = HybridExcelAgent(
        filename=file_path,
        config=config,
        separator=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        over_lap=300
    )

    # Show raw dataframe
    print("\n===== RAW DATAFRAME =====")
    print(agent.sheets)


    # Show vector store info
    print("\n===== VECTOR STORE INFO =====")

    # vector_store = agent.create_text_vectors(
    #     separator=["\n\n", "\n", " ", ""],
    #     chunksize=4000,
    #     overlap=300
    # )

    # print(type(vector_store))