import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# --- 1. Load the model ---
load_dotenv()

# Load the model
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# --- 2. Define the agent state ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # store any of these data types

# --- 3. Define the node functions ---
def process(state: AgentState)->AgentState:
    response = llm.invoke(state["messages"])  

    state["messages"].append(AIMessage(content = response.content))
    print(f"\nAI: {response.content}")
    return state

# --- 4. Define & Build the graph ---
graph = StateGraph(AgentState)
graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
    
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content = user_input))

    # Invoke the agent
    result = agent.invoke({"messages": conversation_history})

    # Update the conversation history and user input for the next iteration
    conversation_history = result["messages"]
    user_input = input("Enter: ")

file_name = "memory_agent.txt"
with open(file_name, "w", encoding = "utf-8") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        """Check whether the message object belongs to either HumanMessage or AIMessage using isinstance(.....)"""
        
        if isinstance(message, HumanMessage):   # Is the current 'message' object an instance of the 'HumanMessage' class.
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):    # Is the current 'message' object an instance of the 'AIMessage' class.
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to memory_agent.txt")
