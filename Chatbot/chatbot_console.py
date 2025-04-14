import os
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from langgraph.graph import StateGraph
import google.generativeai as genai  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = "xxxxxxxxxxxxxxxxxx"

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')


class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = genai.GenerationConfig(
            temperature=1.0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.llm=ChatGoogleGenerativeAI(temperature=0.8,model="gemini-1.5-flash", google_api_key=key,top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=3000)


# Declaring LLMOutputOnlyString Model class
# this is how the LLM is 
# expected to output only a string 
class LLMOutputOnlyString(BaseModel):

    # the LLM is expected to output a strings
    content: str = Field(description="The output can be a plain string")

class QueryStateForString(BaseModel):
    query: str
    output: Optional[str] = None # the LLM is expected to output only as a string.
    rejection_count: int = 0

# Declaring LLMOutput Model class
# this is how the LLM is expected to 
# output the response
class LLMOutput(BaseModel):

    # the LLM is expected to output a strings
    content: Union[str, List[str]] = Field(description="The output can be a plain string or a list of strings")

# Declaring a Final Output class 
# format to be displayed
class QueryState(BaseModel):
    query: str
    output: Optional[Union[str, List[str]]] = None # the LLM is expected to output a string or a list of strings
    human_approval: Optional[str] = None
    next: Optional[str] = None
    rejection_count: int = 0

class PromptTemplates:

    '''
    class for creating prompt Instructions for the LLM
    '''
    
    # creating the input prompt template 
    # as an input to the model
    @classmethod
    def chat_prompt_template(cls):

        try:
            template = """Respond to the following query.
                        Do not Provide any Note or any wrong answers.
                        

                        Attempt: 
                        {attempt}.
                        
                        Format:
                        {format_instructions}

                        Query:
                        {query}
                        """
            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
        
    @classmethod
    def comma_seperated_prompt_template(cls):

        try:
            template = """You are a helpful assistant.

                        Respond to the following query by listing the items as instructed.
                        
                        Attempt: 
                        {attempt}.
                        
                        Format:
                        {format_instructions}

                        Query:
                        {query}
                        """
            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
        
# creating an output parser class
class PydanticOutputs:
    '''
    class for declaring the output parsers
    1. PydanticOutputParser
    2. CommaSeparatedListOutputParser
    '''
    # the LLM should return a 
    # pydantic as response
    @classmethod
    def DictOutputParser(cls,method: None):

        try:
            return PydanticOutputParser(pydantic_object=method)
        except Exception as e:
            return e
        
    # the LLM should return a 
    # list of items
    # seperated by commas
    @classmethod
    def CommaSeperatedParser(cls):

        try:
            return CommaSeparatedListOutputParser()
        except Exception as e:
            return e
    
    @classmethod
    def OtherOutputParser(cls):

        try:
            pass
        except Exception as e:
            return e
    
# declaring a class HumanInTheLoop
# for demonstrating working of HITL 
class HumanInTheLoop(ChatGoogleGENAI): 

    def __init__(self, state_cls,output_state):
        super().__init__()
        self.state_cls = state_cls  # accepts multiple states
        self.opstate = output_state

    # generate output Node
    def generate_output(self, state):
        try:
            inputs = {
                "query": state.query,
                "format_instructions": PydanticOutputs.DictOutputParser(method=self.opstate).get_format_instructions(),
                "attempt": state.rejection_count + 1
            }

            messages = PromptTemplates.chat_prompt_template().format(**inputs)
            response = self.llm.invoke(messages)
            result = PydanticOutputs.DictOutputParser(method=self.opstate).parse(response.content)

            return self.state_cls(
                query=state.query,
                output=result.content,
                rejection_count=state.rejection_count,
                
            )
        except Exception as e:
            print('Error in generate_output: ', e)
            return state

    # final output
    def final_output(self, state):
        try:
            return state
        except Exception as e:
            return state
        
# Define a new class that builds and runs a graph for human-in-the-loop processing
class GraphForHumanInTheLoop(HumanInTheLoop):
    
    # When this class is created, we take two arguments: the state and the output state
    def __init__(self, state: None, opstate: None):
        # We initialize the parent HumanInTheLoop class with the given state and output state
        super().__init__(state_cls=state, output_state=opstate)
        
    # This method builds the actual graph logic
    def build_graph(self):
        try:
            # Create a new graph using the provided state class
            builder = StateGraph(self.state_cls)

            # Add a step (or node) to generate output using an LLM
            builder.add_node("generate_output", self.generate_output)

            # Add a step where the human gives final approval or decision
            builder.add_node("final_decision", self.final_output)

            # Connect the output generation step to the final decision step
            builder.add_edge("generate_output", "final_decision")

            # Connect the final decision step to the end of the graph
            builder.add_edge("final_decision", END)

            # Set the starting point of the graph to be the output generation step
            builder.set_entry_point("generate_output")
            
            # Return the graph that we built
            return builder

        # If there's any error while building the graph, return the error
        except Exception as e:
            return e
        
    # This method actually runs the graph with a user's query
    def execuete_graph(self, user_query: str):
        try:
            # First, build the graph structure
            graph_builder = self.build_graph()

            # Compile the graph so it can be executed
            execuetor = graph_builder.compile()
            
            # Create the starting state using the user's input
            initial_state = QueryState(query=user_query)

            # Run the graph and get the result
            result = execuetor.invoke(initial_state)

            # Return the final result after the graph finishes
            return result

        # If anything goes wrong while running the graph, print the error
        except Exception as e:
            print(f"Fatal error running LangGraph: {e}")


# This class handles running the chatbot and keeping track of user decisions (like rejection or approval)
class Execuetion:

    # When we create an Execuetion object, we give it the user's input, the number of rejections, and whether the user approved the result
    def __init__(self, user_input: None, rejection_count: None, human_approval: None):

        # Save the user's question or message
        self.user_input = user_input

        # Keep track of how many times the user has rejected the chatbot's response
        self.rejection_count = rejection_count

        # Store whether the user approved the last chatbot response ('yes' or 'no')
        self.human_approval = human_approval

        # Create a chatbot graph (a flow of steps) using specific input/output state classes
        self.g = GraphForHumanInTheLoop(state=QueryStateForString, opstate=LLMOutputOnlyString)

    # This function runs the chatbot and gets a response based on the user input
    def get_chatbot_output(self):
        try:
            # Try to run the chatbot and return its result
            return self.g.execuete_graph(user_query=self.user_input)
        except Exception as e:
            # If there's an error, return the error message
            return e

    # Recursive function to handle human approval
    def HITL_handeling(self):
        
        """Recursive function to handle the approval/rejection cycle with dictionary state."""
        
        # Execute the graph and get the result
        chatbot_result = self.get_chatbot_output()
        print(chatbot_result['output'])
        # Ask for human input for approval or rejection
        human_input = input("Enter 'yes' for approval or 'no' for rejection: ").strip().lower()
        
        # Update the dictionary values based on the user input
        if human_input == 'yes':
            self.human_approval = 'yes'
            print("Approval granted. Final Output:")
            return {"query": self.user_input,"reply": chatbot_result['output'] ,"total_rejections": self.rejection_count}  # End recursion and return the result
        
        elif human_input == 'no':
            # Increment rejection count if rejected
            self.rejection_count += 1
            chatbot_result['rejection_count'] = self.rejection_count  # Update rejection count in the dictionary
            chatbot_result['next'] = 'await_human'  # Keep the flow as 'await_human' for the next prompt
            
            print(f"Rejection count: {self.rejection_count}")
            print("Re-running the process for updated output...")
            
            # Re-initialize the graph for another attempt
            g = GraphForHumanInTheLoop(state=QueryStateForString, opstate=LLMOutputOnlyString)
            
            # Recursive call to handle the next attempt
            return self.HITL_handeling()
        
        else:
            # If input is invalid, prompt again
            print("Invalid input. Please enter 'yes' or 'no'.")
            return self.HITL_handeling()
    

# Main function to start the process
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    
    e = Execuetion(user_input=user_input,rejection_count=0,human_approval=None)

    # Start the recursive process
    final_result = e.HITL_handeling()

    # Once approved, print the final result
    if final_result:
        print("Final result after approval:")
        print(final_result)