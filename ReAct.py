from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage  # The foundational class for all message types in LangGraph
# SystemMessage, ToolMessage, HumanMessage, AIMessage, etc. are the child classes of BaseMessage.
from langchain_core.messages import ToolMessage  # Passes data(content and the tool_call_id) back to LLM after it calls a tool.
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages  # It is a specialized state-reducer function that understands the conventions of LangGraph's state representation.
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode  # Node function to implement the tools


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# tool description in a docstring is a must in order for the tool to work.
@tool  # decorater for the tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers together"""
    return a * b

tools = [add, multiply]
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash").bind_tools(tools)  # bind the tools with llm so that it has access to all the tools in the list

# Define the agent node
def model_call(state: AgentState)-> AgentState:
    # We can simply define a single prompt including system detail(what kind of chatmodel) but here we use separate messages packages
    system_prompt = SystemMessage(content = 
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state['messages'])  # add query along with the system prompt
    return {"messages": [response]} # We just update the messages with the response, because the 'add_messages' reducer function handles everything(no overwrite).

# conditional function
def should_continue(state: AgentState)-> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools = tools)  # tool node function
graph.add_node("tools", tool_node)


graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    } 
)  # Go to either tool node or end node
graph.add_edge("tools", "our_agent")  # Circle back to the agent node
app = graph.compile()


def print_stream(stream):   # Handle the streaming output from the compiled graph (app).
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):  # if message is a simpel tuple display that
            print(message)
        else:
            message.pretty_print()  # display in a formatted way rather than a raw, continuous string

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6 also tell a joke")]}
print_stream(app.stream(inputs, stream_mode="values"))  
