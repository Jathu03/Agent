from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

pdf_path = "converted_text.pdf"

# Safety measure for debuggig purposes
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

pages_split = text_splitter.split_documents(pages)

# Chroma database directory
persist_directory = "C:/SJ/Projects/AI/agent_tutorial/langgraph/freecodecamp/chroma_db"
collenction_name = "bool"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Create chroma database using our embedding model
    vector_store = Chroma.from_documents(
        documents = pages_split,
        embedding = embeddings,
        persist_directory = persist_directory,
        collection_name = collenction_name
    )
    print("Created Chroma vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise

# Now we create our retriever
retriever = vector_store.as_retriever(
    search_types = "similarity",
    search_kwargs = {"k":5}
)


@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the PDF."""

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""

    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions. Use the retriever tool available to answer questions. If you need to look up some information
before asking a follow up question, you are allowed to do that!. Please always cite the specific parts of the documents you use in your answer.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Agent
def call_llm(state: AgentState):
    """Function to call llm with the current state."""

    messages = list(state['messages'])
    messages = [SystemMessage(content = system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

# Retriever Agent
def take_actions(state: AgentState):
    """Execute tool calls from the llm's response"""

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provided')}")

        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']}")
            result = "Incorrect tool name, please retry and select tool from the list of available tools"
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Results length: {len(str(result))}")

        # Append the Tool Message
        results.append(ToolMessage(tool_call_id = t['id'], name = t['name'], content = str(result)))
    
    print("Tool Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_actions)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT ===\n")

    while True:
        user_input = input("What is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [HumanMessage(content = user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()