# Human-in-the-Loop LangGraph Chatbot

This project demonstrates a **Human-in-the-Loop (HITL)** workflow using **LangGraph**, **LangChain**, and **LLMs** like Google Gemini (via `ChatGoogleGENAI`) to build an interactive chatbot that involves human validation at every step of output generation.

It supports **dynamic state classes**, multiple **output parsers**, and flexible **prompt templates** — making it easy to plug in both structured and unstructured LLM responses.

---

## 🚀 Features

- Human validation before finalizing LLM output
- Regenerates answers until approved
- Dynamically supports multiple state classes (`QueryState`, `DataQueryState`, etc.)
- Supports multiple parsers:
- `PydanticOutputParser` for structured schema validation
- `CommaSeparatedListOutputParser` for simple lists
- Prompt templates customizable to different formats
- Modular and scalable architecture for future workflows


## How It Works

### 1. The Graph

***You define a LangGraph where nodes are:***

    - `generate_output` — Uses LLM to produce a response
    - `await_human` — Pauses and asks for human validation
    - `rejected` — If rejected, retries generation
    - `final_output` — Final accepted response
    - `route_human_approval` — Edge router based on human feedback

### 2. State Handling

***Each node accepts a `state` object (e.g., `QueryState`) with:***

    - `query` — user input
    - `output` — LLM response
    - `rejection_count` — track number of retries
    - `human_approval` — yes/no response

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

***Supports dynamic states using class injection in `HumanInTheLoop(state_cls, parser)`.***

### 3. Output Parsing

Choose output format:

    - `PydanticOutputParser` → Validates a structured format (`LLMOutput`)
    - `CommaSeparatedListOutputParser` → Parses comma-separated text into a list

    # creating an output parser class
    class PydanticOutputs:
        '''
        class for declaring the output parsers
        1. PydanticOutputParser
        2. CommaSeperatedListOutputParser
        '''


### 4. Prompt Engineering

Prompts are dynamically generated using `PromptTemplates`. You can pass in the output parser's `get_format_instructions()` to guide the LLM to respond in the correct format.

### Dependencies

***Install dependencies:***


    pip install langchain langgraph google-generativeai pydantic

### Notes

    Prompt templates must clearly instruct the LLM to follow output format, especially when using PydanticOutputParser.

    You can switch parsers dynamically to suit different output needs (structured vs list).

    To extend, just create a new State class (e.g. DataQueryState) and plug it into the bot.