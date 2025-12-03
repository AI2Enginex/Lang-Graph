from LLMUtils.LLMConfigs import ChatGoogleGENAI, GeminiConfig, QAState, api_key
from langgraph.graph import StateGraph
from LLMUtils.PromptClass import PromptManager
from LLMUtils.TextProcessing import PrepareText

# ========================== QASYSTEM ============================

class QASystem(PrepareText, ChatGoogleGENAI):
    """Combines text preparation and ChatGoogleGENAI to create a QA system."""

    def __init__(self, file_path: str, config=None,
                 separator=None, chunk_size=None, overlap=None):
        try:
            # Initialize both parents properly (no api_key passed here)
            PrepareText.__init__(self, file_path=file_path,config=config)
            ChatGoogleGENAI.__init__(self, config=config)

            # Create FAISS vector store from document
            self.vector_store = self.create_text_vectors(
                separator=separator,
                chunksize=chunk_size,
                overlap=overlap
            )

            print("QASystem initialized successfully.")

        except Exception as e:
            print(f"Error initializing QASystem: {e}")
            self.vector_store = None

    def retrieve_chunks(self, state: QAState):
        """Retrieve top-k document chunks relevant to the question."""
        try:
            if not self.vector_store:
                print("Vector store not initialized!")
                return state
            docs = self.vector_store.similarity_search(state.question, k=4)
            state.retrieved_chunks = [doc.page_content for doc in docs]
            return state
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return state

    def answer_questions(self, state: QAState):
        """Answer the question based on retrieved chunks using selected prompt template."""
        try:
            if not self.llm:
                print("LLM not initialized!")
                return state

            prompt_manager = PromptManager()
            prompt_template = prompt_manager.get_prompt(state.prompt_type)
            if not prompt_template:
                print("Prompt template not found!")
                return state

            context = "\n\n".join(state.retrieved_chunks)
            prompt = prompt_template.format(context=context, question=state.question)
            response = self.llm.invoke(prompt)

            state.answer = response.content if response else ""
            return state
        except Exception as e:
            print(f"Error answering question: {e}")
            return state


# ========================== GRAPH EXECUTION ============================

class QASystemGraphExecution(QASystem):
    """Builds a LangGraph execution graph to automate QA workflow."""

    def __init__(self, file_path: str, config=None,
                 separator=None, chunk_size=None, overlap=None):
        try:
            super().__init__(file_path=file_path, config=config,
                             separator=separator, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            print(f"Error initializing GraphExecution: {e}")

    def build_graph(self):
        try:
            graph = StateGraph(QAState)
            graph.add_node("retrieve", self.retrieve_chunks)
            graph.add_node("QA", self.answer_questions)
            graph.add_edge("retrieve", "QA")
            graph.set_entry_point("retrieve")
            return graph
        except Exception as e:
            print(f"Error building graph: {e}")
            return None

    def answer(self, question: str, prompt_type: str):
        """Executes the graph to answer the question."""
        try:
            graph_executor = self.build_graph()
            if not graph_executor:
                print("Graph not built correctly!")
                return None

            executor = graph_executor.compile()
            initial_state = {
                "question": question,
                "retrieved_chunks": [],
                "answer": "",
                "prompt_type": prompt_type
            }
            result = executor.invoke(initial_state)
            return result.get("answer", "")
        except Exception as e:
            print(f"Error executing graph: {e}")
            return None


# ========================== MAIN EXECUTION ============================

if __name__ == "__main__":
    try:
        config = GeminiConfig(
            chat_model_name="gemini-2.5-flash",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0.0,
            top_p=0.8,
            top_k=32,
            max_output_tokens=3000,
            generation_max_tokens=8192,
            api_key=api_key  # Set your key here or via environment variable
        )

        file_path = "E:/Lang-Graph/wings_of_fire.pdf"

        question = input("Ask your question here: ")
        prompt_type = input("Choose prompt type (key word extraction / chain of thoughts / verification prompt): ")

        qa_system = QASystemGraphExecution(
            file_path=file_path,
            config=config,
            separator=["\n\n", "\n", " ", ""],
            chunk_size=4000,
            overlap=300
        )

        answer = qa_system.answer(question=question, prompt_type=prompt_type)
        print("\nQuestion:", question)
        print("Answer:", answer)

    except Exception as e:
        print(f"Error in main execution: {e}")
