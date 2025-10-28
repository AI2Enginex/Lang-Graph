
from langchain_core.prompts import PromptTemplate # pyright: ignore[reportMissingImports]

class PromptTemplates:
    """
    Contains different prompt templates for summarization tasks.
    """
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
            Your job is to generate a concise summary of the given document in not more than 200 words.
            Try to give a brief summary by explaining each and every point from the satarting till the end.
            Display the summary as if you are giving a presentation.

            Try to summarize each line in the document.

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
            Combine these summaries into a clear, cohesive final summary in 200 words.
            Do not repeat points; focus on merging and refining.

            Partial Summaries:
            {context}

            Final Summary:
            """
            return PromptTemplate(template=reduce_template.strip(), input_variables=["context"])
        except Exception as e:
            raise e

if __name__ == "__main__":
    pass