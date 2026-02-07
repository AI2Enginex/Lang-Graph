from langchain_core.prompts import PromptTemplate

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

            Carefully analyze the content and provide a clear, well-reasoned answer
            based ONLY on the provided information.

            Do NOT speculate or add information that is not supported by the text.
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
    
    @classmethod
    def summarisation_chains(cls):
        """Direct summarization prompt for smaller documents."""
        try:
            prompt_template = """
            You are given a PDF document.
            Your job is to generate a concise summary of the given document in 15 points.
            Try to cover each point from starting till the end.
            Try to summarize each line in the document.

            Context:
            {context}

            Answer:
            """
            return PromptTemplate(template=prompt_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e
        
    @classmethod
    def summarisation_prompt(cls):
        """
        Direct summarization prompt template for summarizing small documents with Strict Constraints.
        """
        try:
            prompt_template = """
            You are given a PDF document .
            Your job is to generate a concise summary of the given document in not more than 600 words.
            Try to give a brief summary from starting till the end.
            make sure no information is missed.
            start the summary with "In this document..".

            Context:
            {context}

            Answer:
            """
            return PromptTemplate(template=prompt_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

    @classmethod
    def map_prompt(cls):
        """
        Map prompt template: Summarizes each chunk of a document individually.
        """
        try:
            map_template = """
            You are given a part (chunk) of a PDF document.
            Summarize the key points of this chunk, try to summarize each point in brief.
            Avoid adding information not present in the chunk.

            Document Chunk:
            {context}

            Chunk Summary:
            """
            return PromptTemplate(template=map_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

    @classmethod
    def reduce_prompt(cls):
        """
        Reduce prompt template: Combines individual chunk summaries into a cohesive final summary.
        """
        try:
            reduce_template = """
            You are provided with multiple chunk-level summaries of a PDF document.
            Combine these summaries into a clear, cohesive final summary.
            Do not repeat points; focus on merging and refining.
            Also avoid Providing summary in bullet points.

            Partial Summaries:
            {context}

            Final Summary:
            """
            return PromptTemplate(template=reduce_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e
        
    @classmethod
    def reflect_prompt_template(cls,summary: str):

        try:
            prompt = f"""
                Is the following summary complete and concise?
                Answer with YES or NO only.

                Summary:
                {summary}
                """
            return prompt
        except Exception as e:
            return e
    
    @classmethod
    def refine_prompt_template(cls, summary: str):

        try:
            prompt = f"""
                Improve the following summary without increasing its length.
                Do not add new sections or repeat headings.

                Summary:
                {summary}
                """
            return prompt
        except Exception as e:
            return e
    

        
class PromptManager:
    def __init__(self):
        self.prompt_dict = {
            "key_word_extraction": PromptTemplates.key_word_extraction,
            "chain_of_thoughts": PromptTemplates.chain_of_thoughts,
            "verification_prompt": PromptTemplates.verification_prompt
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
