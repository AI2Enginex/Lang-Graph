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
api_key = os.environ['GOOGLE_API_KEY'] = "xxxxxxxx"

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
    human_approval: Optional[str] = None
    next: Optional[str] = None
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


# Decalring a class PromptTemplates
# for creating a instruction and response
# template for the LLM
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
    # dictionary as response
    @classmethod
    def DictOutputParser(cls,method: None):

        try:
            return PydanticOutputParser(pydantic_object=method)
        except Exception as e:
            return e
    
    @classmethod
    def CommaSeperatedParser(cls):

        try:
            return CommaSeparatedListOutputParser()
        except Exception as e:
            return e
    # try using diferent output parser as well
    #Option 2: CommaSeparatedListOutputParser – for simple list outputs
    #return CommaSeparatedListOutputParser()

    # Option 3: StructuredOutputParser – for dicts without full Pydantic models
    # schema = [
    #     ResponseSchema(name="title", description="Headline title"),
    #     ResponseSchema(name="summary", description="Brief summary"),
    # ]
    # return StructuredOutputParser.from_response_schemas(schema

    # Option 4: RegexParser – for parsing consistent text patterns
    # return RegexParser(regex=r"Title: (.*)\nSummary: (.*)", output_keys=["title", "summary"])
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


    '''
    An Example for CommaSeperatedListParser() ---- use only when
    required to generate a list of somma seperated values.

    def generate_output(self, state):
    try:
        parser = PydanticOutputs.CommaSeparatedParser()

        inputs = {
            "query": state.query,
            "format_instructions": parser.get_format_instructions(),
            "attempt": state.rejection_count + 1
        }

        messages = PromptTemplates.comma_seperated_prompt_template().format(**inputs)
        response = self.llm.invoke(messages)
        result = parser.parse(response.content)

        return self.state_cls(
            query=state.query,
            output=result,  # result is already a list of strings
            rejection_count=state.rejection_count,
            next="await_human"
        )
    except Exception as e:
        print('Error in generate_output: ', e)
        return state

    '''
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
                next="await_human"
            )
        except Exception as e:
            print('Error in generate_output: ', e)
            return state

    # await human input node
    def await_human(self, state):
        try:
            print(f"\nGenerated Response:\n{state.output}")
            approval = input("\nDo you approve this response? (yes/no): ").strip().lower()

            return self.state_cls(
                query=state.query,
                output=state.output,
                human_approval=approval,
                rejection_count=state.rejection_count,
                next="final_decision" if approval == "yes" else "rejected"
            )
        except Exception as e:
            return state

    # final output
    def final_output(self, state):
        try:
            return state
        except Exception as e:
            return state

    # rejected and retry
    def rejected(self, state):
        try:
            print("\nAnswer rejected by human. Generating a new response...\n")
            return self.state_cls(
                query=state.query,
                rejection_count=state.rejection_count + 1,
                next="generate_output"
            )
        except Exception as e:
            return state

    # Conditional edge based on human input
    def route_human_approval(self, state):
        try:
            if state.human_approval and state.human_approval.lower().strip() == "yes":
                return "final_decision"
            return "rejected"
        except Exception as e:
            print(f"Error in route_human_approval: {e}")
            return "rejected"

        
class GraphForHumanInTheLoop(HumanInTheLoop):
    
    def __init__(self,state: None,opstate: None):

        super().__init__(state_cls=state,output_state=opstate)
        
        
    def build_graph(self):
        try:
            builder = StateGraph(self.state_cls)

            builder.add_node("generate_output", self.generate_output)
            builder.add_node("await_human", self.await_human)
            builder.add_node("final_decision", self.final_output)
            builder.add_node("rejected", self.rejected)
            builder.add_edge("generate_output", "await_human")

            #Conditional routing with router function
            builder.add_conditional_edges("await_human", self.route_human_approval, {"final_decision": "final_decision","rejected": "rejected"})

            builder.add_edge("rejected", "generate_output")
            builder.add_edge("final_decision", END)

            builder.set_entry_point("generate_output")
            
            
            return builder
        except Exception as e:
            return e
        
    def execuete_graph(self,user_query: str):

        try:
            graph_builder = self.build_graph()
            execuetor = graph_builder.compile()
            
            initial_state = QueryState(query=user_query)
            result = execuetor.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Fatal error running LangGraph: {e}")

        

# Run the Flow
if __name__ == "__main__":
    
    user_input = input("Enter your query: ")
    g = GraphForHumanInTheLoop(state=QueryStateForString,opstate=LLMOutputOnlyString)

    
    chatbot_result = g.execuete_graph(user_query=user_input)
    print(chatbot_result)

    # if the output throws an error
    # code with directly jump to other State
    if chatbot_result['output'] == None:

        g = GraphForHumanInTheLoop(state=QueryState,opstate=LLMOutput)
        chat_result = g.execuete_graph(user_query=user_input)
        print(chatbot_result['output'])
    else:

        print(chatbot_result['output'])


    