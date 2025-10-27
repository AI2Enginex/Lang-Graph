
from LLMConfigs import ChatGoogleGENAI, GeminiConfig, QAState, api_key
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate

from TextProcessing import PrepareText


# ========================== PROMPT TEMPLATES ============================

class PromptTemplates:
    @classmethod
    def key_word_extraction(cls):
        prompt = """
        You are an intelligent assistant.
        Below is the information retrieved from the document:
        {context}
        Answer strictly based on above.
        Question: {question}
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def chain_of_thoughts(cls):
        prompt = """
        You are a thoughtful assistant.
        Here is the document content:
        {context}
        Question: {question}
        Think step by step based ONLY on the provided content.
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def verification_prompt(cls):
        prompt = """
        You are a careful assistant.
        Here is the document content:
        {context}
        Question: {question}
        Verify if the answer is supported by the content.
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

class PromptManager:
    def __init__(self):
        self.prompt_dict = {
            "key word extraction": PromptTemplates.key_word_extraction,
            "chain of thoughts": PromptTemplates.chain_of_thoughts,
            "verification prompt": PromptTemplates.verification_prompt
        }

    def get_prompt(self, name):
        try:
            func = self.prompt_dict.get(name)
            if not func:
                raise ValueError(f"Prompt '{name}' not found!")
            return func()
        except Exception as e:
            print(f"Error retrieving prompt: {e}")
            return None



#

# ========================== QASYSTEM ============================

class QASystem(PrepareText, ChatGoogleGENAI):
    """
    Combines text preparation and ChatGoogleGENAI to create a QA system.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str,
                 separator=None, chunk_size=None, overlap=None):
        try:
            PrepareText.__init__(self, file_path=file_path, config=config, api_key=api_key)
            ChatGoogleGENAI.__init__(self, config=config, api_key=api_key)
            self.vector_store = self.create_text_vectors(separator, chunk_size, overlap)
        except Exception as e:
            print(f"Error initializing QASystem: {e}")
            self.vector_store = None

    def retrieve_chunks(self, state: QAState):
        """
        Retrieve top-k document chunks relevant to the question.
        """
        try:
            if not self.vector_store:
                print("Vector store not initialized!")
                return state
            docs = self.vector_store.similarity_search(state.question, k=4)
            retrieved_chunks = [doc.page_content for doc in docs]
            state.retrieved_chunks = retrieved_chunks
            return state
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return state

    def answer_questions(self, state: QAState):
        """
        Answer the question based on retrieved chunks using selected prompt template.
        """
        try:
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
    """
    Builds a LangGraph execution graph to automate QA workflow.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str,
                 separator=None, chunk_size=None, overlap=None):
        try:
            super().__init__(file_path=file_path, config=config, api_key=api_key,
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
        """
        Executes the graph to answer the question using LangGraph flow.
        """
        try:
            graph_executor = self.build_graph()
            if not graph_executor:
                print("Graph not built correctly!")
                return None

            executor = graph_executor.compile()
            initial_state = {"question": question, "retrieved_chunks": [], "answer": "", "prompt_type": prompt_type}
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
            embedding_model_name="models/gemini-embedding-001",
            temperature=0,
            top_p=0.8,
            top_k=32,
            max_output_tokens=3000,
            generation_max_tokens=8192
        )

        file_path = "E:/Lang-Graph/RILAGM.pdf"

        question = input("Ask your question here: ")
        prompt_type = input("Choose prompt type (key word extraction / chain of thoughts / verification prompt): ")

        qa_system = QASystemGraphExecution(
            file_path=file_path,
            config=config,
            api_key=api_key,
            separator="\n\n",
            chunk_size=4000,
            overlap=300
        )

        answer = qa_system.answer(question=question, prompt_type=prompt_type)
        print("\nQuestion:", question)
        print("Answer:", answer)
    except Exception as e:
        print(f"Error in main execution: {e}")
